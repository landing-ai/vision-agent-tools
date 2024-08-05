from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

VideoNumpy = Annotated[npt.NDArray[DType], Literal["N", "N", "N", 3]]
