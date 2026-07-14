import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections per camera.
    Broadcasts metadata frames to all connected clients.
    """

    def __init__(self):
        # camera_id → set of connected WebSocket clients
        self._connections: Dict[int, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, camera_id: int, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            if camera_id not in self._connections:
                self._connections[camera_id] = set()
            self._connections[camera_id].add(websocket)
        logger.info(f"WS client connected to camera {camera_id} ({len(self._connections[camera_id])} total)")

    async def disconnect(self, camera_id: int, websocket: WebSocket):
        async with self._lock:
            if camera_id in self._connections:
                self._connections[camera_id].discard(websocket)
                if not self._connections[camera_id]:
                    del self._connections[camera_id]
        logger.info(f"WS client disconnected from camera {camera_id}")

    async def broadcast(self, camera_id: int, data: dict):
        """Broadcast metadata to all clients watching this camera."""
        if camera_id not in self._connections:
            return

        dead = set()
        message = json.dumps(data, default=str)

        for ws in list(self._connections.get(camera_id, set())):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)

        if dead:
            async with self._lock:
                if camera_id in self._connections:
                    self._connections[camera_id] -= dead

    async def broadcast_alert(self, alert_data: dict):
        """Broadcast a source-level alert to all connected clients."""
        message = json.dumps({"type": "alert", **alert_data}, default=str)
        for cam_id, clients in list(self._connections.items()):
            dead = set()
            for ws in list(clients):
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            if dead:
                async with self._lock:
                    if cam_id in self._connections:
                        self._connections[cam_id] -= dead

    def get_connection_count(self, camera_id: int) -> int:
        return len(self._connections.get(camera_id, set()))

    def get_total_connections(self) -> int:
        return sum(len(v) for v in self._connections.values())


websocket_manager = WebSocketManager()
