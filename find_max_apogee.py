import json
import sys

max_apogee = 0
max_sim = -1
dir_path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/monte_carlo_20250710_093951/simulation_results/'
for i in range(100):
    try:
        with open(f"{dir_path}/sim_{i}.json") as f:
            data = json.load(f)
            apo = data["apogee_altitude"]
            if apo > max_apogee:
                max_apogee = apo
                max_sim = i
    except:
        pass
print(f"Max apogee in sim_{max_sim}.json: {max_apogee}") 