/**
 * g1_bridge.cpp — pybind11 C++ DDS bridge for Unitree G1.
 *
 * Wraps unitree_sdk2 C++ DDS publish/subscribe so that all realtime
 * communication happens in native threads (<0.5 ms), while Python
 * calls simple get/set functions with zero serialisation overhead.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/dds_wrapper/common/crc.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

namespace py = pybind11;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

// ---- Constants matching standalone_standing.py ----
static constexpr int NUM_JOINTS = 29;
static constexpr int NUM_MOTORS = 35;
static constexpr uint8_t MODE_PR = 0;
static constexpr uint8_t MODE_MACHINE = 5;
static constexpr int PUBLISH_HZ = 500;
static constexpr float POS_STOP_F = 2.146e9f;
static constexpr float VEL_STOP_F = 16000.0f;
static constexpr float KD_DAMPING = 8.0f;

class G1Bridge {
public:
    explicit G1Bridge(const std::string& network_interface)
        : publish_running_(false)
        , state_received_(false)
        , has_target_(false)
        , damping_requested_(false)
    {
        // Zero state buffers
        std::memset(qpos_, 0, sizeof(qpos_));
        std::memset(qvel_, 0, sizeof(qvel_));
        std::memset(quat_, 0, sizeof(quat_));
        std::memset(ang_vel_, 0, sizeof(ang_vel_));
        std::memset(wireless_remote_, 0, sizeof(wireless_remote_));
        mode_machine_ = 0;

        // Zero target buffers
        std::memset(target_pos_, 0, sizeof(target_pos_));
        std::memset(target_kp_, 0, sizeof(target_kp_));
        std::memset(target_kd_, 0, sizeof(target_kd_));

        // DDS init
        unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);

        // Subscriber
        state_sub_ = std::make_unique<ChannelSubscriber<LowState_>>("rt/lowstate");
        state_sub_->InitChannel(
            [this](const void* msg) { this->lowstate_callback(msg); }, 10);

        // Publisher
        cmd_pub_ = std::make_unique<ChannelPublisher<LowCmd_>>("rt/lowcmd");
        cmd_pub_->InitChannel();

        // Build default command
        init_default_cmd();
    }

    ~G1Bridge() {
        stop_publish();
    }

    // Block until first LowState arrives (or timeout)
    bool wait_for_state(double timeout_sec) {
        auto deadline = std::chrono::steady_clock::now()
            + std::chrono::duration<double>(timeout_sec);
        while (!state_received_.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline)
                return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return true;
    }

    // Return (qpos[29], qvel[29], quat[4], ang_vel[3]) as numpy arrays
    py::tuple get_state() {
        // Explicit shape + strides to avoid stride=0 bug
        constexpr ssize_t fs = sizeof(float);
        py::array_t<float> py_qpos({(ssize_t)NUM_JOINTS}, {fs});
        py::array_t<float> py_qvel({(ssize_t)NUM_JOINTS}, {fs});
        py::array_t<float> py_quat({(ssize_t)4}, {fs});
        py::array_t<float> py_ang_vel({(ssize_t)3}, {fs});

        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            std::memcpy(py_qpos.mutable_data(), qpos_, sizeof(qpos_));
            std::memcpy(py_qvel.mutable_data(), qvel_, sizeof(qvel_));
            std::memcpy(py_quat.mutable_data(), quat_, sizeof(quat_));
            std::memcpy(py_ang_vel.mutable_data(), ang_vel_, sizeof(ang_vel_));
        }

        return py::make_tuple(py_qpos, py_qvel, py_quat, py_ang_vel);
    }

    // Return wireless_remote bytes (40 bytes)
    py::bytes get_wireless_remote() {
        char buf[40];
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            std::memcpy(buf, wireless_remote_, 40);
        }
        return py::bytes(buf, 40);
    }

    // Return current mode_machine
    uint8_t get_mode_machine() {
        std::lock_guard<std::mutex> lk(state_mutex_);
        return mode_machine_;
    }

    // Set target positions, kp, kd (Python writes, C++ 500Hz reads)
    void set_target(py::array_t<float> target,
                    py::array_t<float> kp,
                    py::array_t<float> kd) {
        auto t = target.unchecked<1>();
        auto k = kp.unchecked<1>();
        auto d = kd.unchecked<1>();

        if (t.shape(0) < NUM_JOINTS || k.shape(0) < NUM_JOINTS || d.shape(0) < NUM_JOINTS) {
            throw std::runtime_error("set_target: arrays must have >= 29 elements");
        }

        {
            std::lock_guard<std::mutex> lk(target_mutex_);
            for (int i = 0; i < NUM_JOINTS; ++i) {
                target_pos_[i] = t(i);
                target_kp_[i] = k(i);
                target_kd_[i] = d(i);
            }
        }
        has_target_.store(true, std::memory_order_release);
        damping_requested_.store(false, std::memory_order_release);
    }

    // Lock joints at current position with given gains
    void lock_joints() {
        std::lock_guard<std::mutex> lk_s(state_mutex_);
        std::lock_guard<std::mutex> lk_t(target_mutex_);
        std::memcpy(target_pos_, qpos_, sizeof(qpos_));
        // Use default KP/KD — caller should set_target with proper gains afterwards
        // For lock, zero out kp/kd and let publish_loop use the values set here
        // Actually we need reasonable gains to hold position — use zeros and let
        // Python provide them. But the plan says lock_joints uses current qpos as target.
        // We'll store qpos and mark has_target; Python must have already called set_target
        // with proper kp/kd, or we use stored kp/kd.
        has_target_.store(true, std::memory_order_release);
        damping_requested_.store(false, std::memory_order_release);
    }

    // Request damping mode
    void set_damping() {
        damping_requested_.store(true, std::memory_order_release);
    }

    // Start 500Hz publish thread
    void start_publish() {
        if (publish_running_.load())
            return;
        publish_running_.store(true, std::memory_order_release);
        publish_thread_ = std::thread(&G1Bridge::publish_loop, this);
    }

    // Stop publish thread
    void stop_publish() {
        publish_running_.store(false, std::memory_order_release);
        if (publish_thread_.joinable())
            publish_thread_.join();
    }

private:
    void init_default_cmd() {
        // Zero-init the command
        cmd_ = LowCmd_();
        cmd_.mode_pr() = MODE_PR;
        cmd_.mode_machine() = MODE_MACHINE;

        for (int i = 0; i < NUM_MOTORS; ++i) {
            auto& mc = cmd_.motor_cmd()[i];
            mc.mode() = 0x01;
            if (i < NUM_JOINTS) {
                mc.q() = 0.0f;
                mc.kp() = 0.0f;
                mc.dq() = 0.0f;
                mc.kd() = KD_DAMPING;
                mc.tau() = 0.0f;
            } else {
                mc.q() = POS_STOP_F;
                mc.kp() = 0.0f;
                mc.dq() = VEL_STOP_F;
                mc.kd() = 0.0f;
                mc.tau() = 0.0f;
            }
        }
    }

    void lowstate_callback(const void* msg) {
        const LowState_ state = *(const LowState_*)msg;

        std::lock_guard<std::mutex> lk(state_mutex_);
        for (int i = 0; i < NUM_JOINTS; ++i) {
            qpos_[i] = state.motor_state()[i].q();
            qvel_[i] = state.motor_state()[i].dq();
        }
        quat_[0] = state.imu_state().quaternion()[0];
        quat_[1] = state.imu_state().quaternion()[1];
        quat_[2] = state.imu_state().quaternion()[2];
        quat_[3] = state.imu_state().quaternion()[3];
        ang_vel_[0] = state.imu_state().gyroscope()[0];
        ang_vel_[1] = state.imu_state().gyroscope()[1];
        ang_vel_[2] = state.imu_state().gyroscope()[2];

        std::memcpy(wireless_remote_, state.wireless_remote().data(),
                     std::min<size_t>(40, state.wireless_remote().size()));
        mode_machine_ = state.mode_machine();

        state_received_.store(true, std::memory_order_release);
    }

    void publish_loop() {
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / PUBLISH_HZ));

        auto next_tick = clock::now();

        while (publish_running_.load(std::memory_order_acquire)) {
            if (damping_requested_.load(std::memory_order_acquire)) {
                // Damping: zero position gains, only kd
                for (int i = 0; i < NUM_MOTORS; ++i) {
                    auto& mc = cmd_.motor_cmd()[i];
                    mc.mode() = 0x01;
                    mc.q() = 0.0f;
                    mc.kp() = 0.0f;
                    mc.dq() = 0.0f;
                    mc.kd() = KD_DAMPING;
                    mc.tau() = 0.0f;
                }
            } else if (has_target_.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lk(target_mutex_);
                for (int i = 0; i < NUM_JOINTS; ++i) {
                    auto& mc = cmd_.motor_cmd()[i];
                    mc.mode() = 0x01;
                    mc.q() = target_pos_[i];
                    mc.kp() = target_kp_[i];
                    mc.dq() = 0.0f;
                    mc.kd() = target_kd_[i];
                    mc.tau() = 0.0f;
                }
            }

            cmd_.mode_machine() = MODE_MACHINE;
            cmd_.crc() = crc32_core((uint32_t*)&cmd_, (sizeof(cmd_) >> 2) - 1);
            cmd_pub_->Write(cmd_);

            next_tick += period;
            auto now = clock::now();
            if (next_tick > now) {
                std::this_thread::sleep_until(next_tick);
            } else {
                // Missed deadline — reset to avoid burst of catch-up publishes
                next_tick = now + period;
            }
        }
    }

    // DDS handles
    std::unique_ptr<ChannelPublisher<LowCmd_>>   cmd_pub_;
    std::unique_ptr<ChannelSubscriber<LowState_>> state_sub_;

    // Command template
    LowCmd_ cmd_;

    // State buffer (callback writes, Python reads)
    std::mutex state_mutex_;
    float qpos_[NUM_JOINTS]{};
    float qvel_[NUM_JOINTS]{};
    float quat_[4]{};
    float ang_vel_[3]{};
    uint8_t wireless_remote_[40]{};
    uint8_t mode_machine_{0};
    std::atomic<bool> state_received_;

    // Target buffer (Python writes, publish thread reads)
    std::mutex target_mutex_;
    float target_pos_[NUM_JOINTS]{};
    float target_kp_[NUM_JOINTS]{};
    float target_kd_[NUM_JOINTS]{};
    std::atomic<bool> has_target_;
    std::atomic<bool> damping_requested_;

    // Publish thread
    std::thread publish_thread_;
    std::atomic<bool> publish_running_;
};

// ---- pybind11 module ----
PYBIND11_MODULE(g1_bridge_sdk, m) {
    m.doc() = "G1 C++ DDS bridge for unitree_sdk2";

    py::class_<G1Bridge>(m, "G1Bridge")
        .def(py::init<const std::string&>(), py::arg("network_interface"))
        .def("wait_for_state", &G1Bridge::wait_for_state,
             py::arg("timeout_sec") = 5.0,
             py::call_guard<py::gil_scoped_release>(),
             "Block until first LowState arrives (returns False on timeout)")
        .def("get_state", &G1Bridge::get_state,
             "Return (qpos[29], qvel[29], quat[4], ang_vel[3]) numpy arrays")
        .def("get_wireless_remote", &G1Bridge::get_wireless_remote,
             "Return 40-byte wireless remote data")
        .def("get_mode_machine", &G1Bridge::get_mode_machine,
             "Return current mode_machine value")
        .def("set_target", &G1Bridge::set_target,
             py::arg("target"), py::arg("kp"), py::arg("kd"),
             "Set target positions and PD gains (29 elements each)")
        .def("lock_joints", &G1Bridge::lock_joints,
             "Lock joints at current position")
        .def("set_damping", &G1Bridge::set_damping,
             "Switch to damping mode (all motors kd-only)")
        .def("start_publish", &G1Bridge::start_publish,
             "Start 500Hz command publish thread")
        .def("stop_publish", &G1Bridge::stop_publish,
             py::call_guard<py::gil_scoped_release>(),
             "Stop publish thread");
}
