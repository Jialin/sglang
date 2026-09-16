# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import importlib.util
import logging
import os
import time
from typing import List

import requests

logger = logging.getLogger(__name__)


def diagnose_remote_weight(stage, model, metadata=None, include_values=False):
    """DIAGNOSTIC ONLY: trace the first corrupted norm without changing copies."""
    import torch

    name = "model.layers.0.input_layernorm.weight"
    weight = model.get_parameter(name)
    pointer = weight.data_ptr()
    record = {
        "stage": stage,
        "pid": os.getpid(),
        "name": name,
        "device": str(weight.device),
        "pointer": pointer,
        "storage_pointer": weight.untyped_storage().data_ptr(),
        "storage_bytes": weight.untyped_storage().nbytes(),
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "metadata": metadata,
    }
    for segment in torch.cuda.memory.memory_snapshot():
        if segment["address"] <= pointer < segment["address"] + segment["total_size"]:
            record["allocation"] = {
                "address": segment["address"],
                "size": segment["total_size"],
                "offset": pointer - segment["address"],
                "blocks": [
                    {
                        key: block[key]
                        for key in ("address", "size", "requested_size", "state")
                        if key in block
                    }
                    for block in segment["blocks"]
                    if block["address"] <= pointer < block["address"] + block["size"]
                ],
            }
            break
    # Do not synchronize before the transfer: that could mask ordering bugs.
    if include_values:
        record["values"] = weight.detach().reshape(-1)[:10].cpu().float().tolist()
    print("REMOTE_WEIGHT_DIAGNOSTIC", record, flush=True)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


def get_remote_instance_transfer_engine_info_per_rank(seed_url: str, rank: int):
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
            },
        )

        if response.status_code == 200:
            data = response.json()

            if "remote_instance_transfer_engine_info" in data:
                return data["remote_instance_transfer_engine_info"]
            else:
                logger.error(
                    "Failed to get `remote_instance_transfer_engine_info` in response."
                )
                return None, None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None, None


def register_memory_region(model, transfer_engine):
    if importlib.util.find_spec("torch") is None:
        return register_memory_region_v1(model, transfer_engine)
    else:
        return register_memory_region_v2(model, transfer_engine)


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    for name, weight in model.named_parameters():
        ret = transfer_engine.register_memory(
            weight.data_ptr(), weight.numel() * weight.element_size()
        )
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight {name}, error: {ret}"
            )
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_addr_set = set()
    for name, weight in model.named_parameters():
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )
        weight_addr_set.add(weight.data_ptr())

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        if torch.version.hip:
            # DIAGNOSTIC ONLY: HIP IPC requires the allocation's base pointer.
            address = segment["address"]
            size = segment["total_size"]
            if any(address <= pointer < address + size for pointer in weight_addr_set):
                weight_blocks_for_reg_mr.append((address, size))
            continue
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            # Only register active allocated memory blocks that hold weights.
            if state == "active_allocated":
                if address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    # Register merged memory blocks that hold weights.
    for weight_block in weight_blocks_for_reg_mr:
        address, size = weight_block
        ret = transfer_engine.register_memory(address, size)
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight block at address {address} with size {size}, error: {ret}"
            )

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    diagnostic_name = "model.layers.0.input_layernorm.weight"
    diagnostic_pointer = weight_mr_dict[diagnostic_name][0]
    diagnose_remote_weight(
        "registered",
        model,
        {
            "published": weight_mr_dict[diagnostic_name],
            "registered_regions": [
                region
                for region in weight_blocks_for_reg_mr
                if region[0] <= diagnostic_pointer < region[0] + region[1]
            ],
        },
    )
    return weight_mr_dict
