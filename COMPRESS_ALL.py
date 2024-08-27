import subprocess
script_path = "Script/test_sklist-cmp.sh"
data_paths = ["DataSets/USE/BiRd", "DataSets/USE/EsCo", "DataSets/USE/ArTh", "DataSets/USE/HuMA"]

for data_path in data_paths:
    process = subprocess.Popen(["bash", script_path, data_path, "0", "32"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
    if process.returncode != 0:
        print(f"Error executing {script_path} {data_path}: {stderr.decode()}")
        # break