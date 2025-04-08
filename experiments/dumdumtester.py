import subprocess
import os

def test(window_size, tolerance, delta_buy, delta_sell):

    path = "jmerle.py"
    with open(path, "r") as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if "RAINFOREST_RESIN" in line and "JemerleStrategy" in line:
            new_lines.append(f'"RAINFOREST_RESIN": JemerleStrategy("RAINFOREST_RESIN", 50, window_size={window_size}, tolerance={tolerance}, delta_buy={delta_buy}, delta_sell={delta_sell})\n')
        else:
            new_lines.append(line)

    with open(path, "w") as file:
        file.write("".join(new_lines))

    venv_python = ".venv/bin/python"  # Path to Python in the virtual environment
    command = [venv_python, "-m", "prosperity3bt", "jmerle.py", "0"]
    result = subprocess.run(command, capture_output=True, text=True)

    return int(result.stdout.split('\n')[2].split(':')[1].replace(",", ""))


best_score = 0
for window_size in range(1, 5):
    for tolerance in (0.2, 0.4, 0.6, .8):
        for delta_buy in (0, 1, 2):
            for delta_sell in (0, 1, 2):
                score = test(window_size, tolerance, delta_buy, delta_sell)
                if score > best_score:
                    best_score = score
                    print(f"window_size={window_size}, tolerance={tolerance}, score={score}, delta_buy={delta_buy}, delta_sell={delta_sell}")

