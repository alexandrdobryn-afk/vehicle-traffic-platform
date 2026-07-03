"""Fail a container build/startup check when critical runtime packages are incomplete."""

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from uvicorn.main import main as uvicorn_main

import cv2
import onnxruntime
import torch
import uvicorn


assert FastAPI is not None
assert AsyncSession is not None
assert callable(uvicorn_main)
assert callable(cv2.imread)
assert hasattr(torch, "cuda")
assert callable(onnxruntime.get_available_providers)
assert getattr(uvicorn, "__version__", None)

print("IMAGE_INTEGRITY_OK")
