"""
WebSocket server untuk menerima detection events dan broadcast ke dashboard clients.

Server ini menerima events dari detection systems dan mem-broadcast ke semua
dashboard clients yang terhubung.
"""

import asyncio
import json
import logging
from typing import Set
import websockets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Store connected dashboard clients
dashboard_clients: Set = set()

# Store connected detection devices
detection_devices: Set = set()


async def handle_dashboard_client(websocket, path: str):
    """Handle dashboard client connections with smart logging."""
    # Check if this is the first client
    is_first_client = len(dashboard_clients) == 0

    dashboard_clients.add(websocket)
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"

    # Only log when first client connects
    if is_first_client:
        logger.info(f"First dashboard client connected: {client_id}")
        logger.info(f"Total dashboard clients: {len(dashboard_clients)}")
    else:
        logger.debug(f"Dashboard client connected: {client_id}")
        logger.debug(f"Total dashboard clients: {len(dashboard_clients)}")

    try:
        # Keep connection alive and wait for disconnect
        async for message in websocket:
            # Dashboard clients don't send messages, only receive
            logger.debug(f"Received message from dashboard client: {message}")
    except websockets.exceptions.ConnectionClosed:
        # Only log if clients are still connected
        if len(dashboard_clients) > 1:
            logger.debug(f"Dashboard client disconnected: {client_id}")
    except Exception as e:
        # Always log errors when clients are connected
        if len(dashboard_clients) > 0:
            logger.error(f"Error in dashboard client {client_id}: {e}")
    finally:
        dashboard_clients.remove(websocket)

        # Only log when last client disconnects
        if len(dashboard_clients) == 0:
            logger.info(f"Last dashboard client disconnected: {client_id}")
            logger.info(f"Total dashboard clients: {len(dashboard_clients)}")
        else:
            logger.debug(f"Total dashboard clients: {len(dashboard_clients)}")


async def handle_detection_device(websocket, path: str):
    """Handle detection device connections and broadcast events to dashboards."""
    detection_devices.add(websocket)
    device_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"

    # Only log device connections when dashboard clients are connected
    if len(dashboard_clients) > 0:
        logger.info(f"Detection device connected: {device_id}")
        logger.info(f"Total detection devices: {len(detection_devices)}")
    else:
        logger.debug(f"Detection device connected: {device_id} (no dashboard clients)")

    try:
        async for message in websocket:
            try:
                # Parse and validate message
                data = json.loads(message)

                # Route based on message type
                message_type = data.get("type", "unknown")

                if message_type == "frame":
                    # Video frame message
                    required_fields = [
                        "type",
                        "timestamp",
                        "source_id",
                        "frame_number",
                        "frame",
                    ]
                    if not all(field in data for field in required_fields):
                        logger.warning(
                            f"Invalid frame message from {device_id}: missing required fields"
                        )
                        continue

                    logger.debug(
                        f"Received video frame from {data.get('source_id')}, frame {data.get('frame_number')}"
                    )

                elif message_type == "detection":
                    # Detection event - validate required fields
                    required_fields = [
                        "type",
                        "timestamp",
                        "source_id",
                        "current_count",
                        "frame_number",
                        "event_type",
                    ]
                    if not all(field in data for field in required_fields):
                        logger.warning(
                            f"Invalid detection message from {device_id}: missing required fields"
                        )
                        continue

                    logger.debug(
                        f"Received detection from {data.get('source_id')}: count={data.get('current_count')}"
                    )

                else:
                    # Unknown message type - try to handle for backward compatibility
                    if "frame" in data:
                        logger.debug(
                            f"Received legacy frame message from {data.get('source_id')}"
                        )
                    elif "current_count" in data:
                        logger.debug(
                            f"Received legacy detection message from {data.get('source_id')}"
                        )
                    else:
                        logger.warning(
                            f"Unknown message type '{message_type}' from {device_id}"
                        )
                        continue

                # Broadcast to all dashboard clients
                if dashboard_clients:
                    await asyncio.gather(
                        *[client.send(message) for client in dashboard_clients],
                        return_exceptions=True,
                    )
                    logger.debug(
                        f"Broadcasted {message_type} to {len(dashboard_clients)} dashboard clients"
                    )
                else:
                    logger.debug(
                        "No dashboard clients connected, message not broadcasted"
                    )

            except json.JSONDecodeError:
                # Only log errors when clients are connected
                if len(dashboard_clients) > 0:
                    logger.error(f"Invalid JSON from {device_id}")
            except Exception as e:
                # Only log errors when clients are connected
                if len(dashboard_clients) > 0:
                    logger.error(f"Error processing message from {device_id}: {e}")

    except websockets.exceptions.ConnectionClosed:
        # Only log disconnections when dashboard clients are connected
        if len(dashboard_clients) > 0:
            logger.info(f"Detection device disconnected: {device_id}")
    finally:
        detection_devices.remove(websocket)
        if len(dashboard_clients) > 0:
            logger.info(f"Total detection devices: {len(detection_devices)}")


async def router(websocket):
    """Route connections based on path."""
    try:
        path = websocket.request.path
        logger.info(f"New connection attempt to path: {path}")

        if path == "/ws":
            # Dashboard client connection
            await handle_dashboard_client(websocket, path)
        elif path == "/device":
            # Detection device connection
            await handle_detection_device(websocket, path)
        else:
            logger.warning(f"Unknown path: {path}")
            await websocket.close()
    except Exception as e:
        logger.error(f"Error in router: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Start WebSocket server."""
    host = "0.0.0.0"
    port = 8001

    logger.info(f"Starting WebSocket server on {host}:{port}")
    logger.info(f"Dashboard clients connect to: ws://{host}:{port}/ws")
    logger.info(f"Detection devices connect to: ws://{host}:{port}/device")

    async with websockets.serve(router, host, port):
        logger.info("WebSocket server started successfully")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
