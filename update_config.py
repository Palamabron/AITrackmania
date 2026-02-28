import json

with open("C:/Users/szulc/TmrlData/config/config.json") as f:
    config = json.load(f)

config["ENV"]["REWARD_CONFIG"]["PROXIMITY_REWARD_SHAPING"] = 0.1
config["ENV"]["REWARD_CONFIG"]["TIME_PENALTY_PER_STEP"] = 0.0005

with open("C:/Users/szulc/TmrlData/config/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("config.json updated!")
