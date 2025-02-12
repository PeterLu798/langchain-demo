from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class Action(BaseModel):
    name: str = Field(description="Tool name")
    args: Optional[Dict[str, Any]] = Field(description="Tool input arguments, containing arguments names and values")

    def __str__(self):
        ret = f"Action(name={self.name}"
        if self.args:
            for k, v in self.args.items():
                ret += f", {k}={v}"
        ret += ")"
        return ret
