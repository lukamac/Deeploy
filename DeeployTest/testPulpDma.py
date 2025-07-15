import itertools
import subprocess

# input shape, tile shape, node count, data type
test_shapes_and_more = [
    ((10, 10), (10, 10), 1, "uint8_t"),
    ((10, 10), (10, 4), 1, "uint8_t"),
    ((10, 10), (10, 4), 1, "uint16_t"),
    ((10, 10), (10, 4), 1, "uint32_t"),
    ((10, 10), (3, 4), 1, "uint32_t"),
    ((10, 10), (3, 4), 2, "uint32_t"),
    ((10, 10, 10), (2, 3, 4), 1, "uint8_t"),
    ((10, 10, 10, 10), (2, 3, 5, 4), 1, "uint8_t"),
    ((10, 10, 10, 10), (2, 3, 5, 4), 1, "uint32_t"),
    ((10, 10, 10, 10, 10), (2, 3, 5, 7, 4), 1, "uint8_t"),
]

is_doublebuffers = [True, False]
defaultMemLevels = ["L2", "L3"]

for test_shape, is_db, defMemLvl in itertools.product(test_shapes_and_more, is_doublebuffers, defaultMemLevels):
    input_shape, tile_shape, node_count, data_type = test_shape

    cfg_str = f"""
    - input shape: {input_shape}
    - tile shape: {tile_shape}
    - node count: {node_count}
    - data type: {data_type}
    - doublebuffering: {is_db}
    - default memory level: {defMemLvl}
    """

    print("testPulpDma: Testing pulp DMA with followig configuration:" + cfg_str)

    cmd = ["python testRunner_pulpDma.py", "-t testPulpDma", "-DNUM_CORES=8"]
    cmd.append(f"--input-shape {' '.join(str(x) for x in input_shape)}")
    cmd.append(f"--tile-shape {' '.join(str(x) for x in tile_shape)}")
    cmd.append(f"--node-count {node_count}")
    cmd.append(f"--type {data_type}")
    cmd.append(f"--defaultMemLevel {defMemLvl}")
    if is_db:
        cmd.append("--doublebuffer")

    full_cmd = " ".join(cmd)

    print(f"Running command:\n{full_cmd}\n")

    try:
        subprocess.run(full_cmd, shell = True, check = True)
    except subprocess.CalledProcessError:
        print("testPulpDma: Failed test:" + cfg_str)
        print(f"Rerun with command:\n{full_cmd}")
        exit(-1)
