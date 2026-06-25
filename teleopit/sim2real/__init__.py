__all__ = [
    "Sim2RealRuntime",
    "UnitreeG1Robot",
    "UnitreeRemote",
    "Button",
]


def __getattr__(name: str):
    if name == "Sim2RealRuntime":
        from teleopit.sim2real.mp import Sim2RealRuntime

        return Sim2RealRuntime
    if name == "UnitreeG1Robot":
        from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

        return UnitreeG1Robot
    if name in ("UnitreeRemote", "Button"):
        from teleopit.sim2real.remote import Button, UnitreeRemote

        return {"UnitreeRemote": UnitreeRemote, "Button": Button}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
