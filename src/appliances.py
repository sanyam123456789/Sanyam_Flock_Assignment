"""Appliance names per channel (CLEAN_READ_ME_081116.txt, as known at the start of the study).
Used only to NAME other loads in the failure analysis - never as model input."""

NAMES = {
    1: ["Fridge", "Freezer (1)", "Freezer (2)", "Washer dryer", "Washing machine", "Dishwasher", "Computer", "Television site", "Electric heater"],
    2: ["Fridge-freezer", "Washing machine", "Dishwasher", "Television site", "Microwave", "Toaster", "Hi-fi", "Kettle", "Overhead fan"],
    3: ["Toaster", "Fridge-freezer", "Freezer", "Tumble dryer", "Dishwasher", "Washing machine", "Television site", "Microwave", "Kettle"],
    4: ["Fridge", "Freezer", "Fridge-freezer", "Washing machine (1)", "Washing machine (2)", "Desktop computer", "Television site", "Microwave", "Kettle"],
    5: ["Fridge-freezer", "Tumble dryer", "Washing machine", "Dishwasher", "Desktop computer", "Television site", "Microwave", "Kettle", "Toaster"],
    6: ["Freezer", "Washing machine", "Dishwasher", "MJY computer", "TV/satellite", "Microwave", "Kettle", "Toaster", "PGM computer"],
    7: ["Fridge", "Freezer (1)", "Freezer (2)", "Tumble dryer", "Washing machine", "Dishwasher", "Television site", "Toaster", "Kettle"],
    8: ["Fridge", "Freezer", "Washer dryer", "Washing machine", "Toaster", "Computer", "Television site", "Microwave", "Kettle"],
    9: ["Fridge-freezer", "Washer dryer", "Washing machine", "Dishwasher", "Television site", "Microwave", "Kettle", "Hi-fi", "Electric heater"],
    10: ["Magimix (blender)", "Toaster", "Chest freezer", "Fridge-freezer", "Washing machine", "Dishwasher", "Television site", "Microwave", "K mix"],
    11: ["Fridge", "Fridge-freezer", "Washing machine", "Dishwasher", "Computer site", "Microwave", "Kettle", "Router", "Hi-fi"],
    12: ["Fridge-freezer", "???", "???", "Computer site", "Microwave", "Kettle", "Toaster", "Television", "???"],
    13: ["Television site", "Freezer", "Washing machine", "Dishwasher", "???", "Network site", "Microwave", "Microwave", "Kettle"],
    15: ["Fridge-freezer", "Tumble dryer", "Washing machine", "Dishwasher", "Computer site", "Television site", "Microwave", "Hi-fi", "Toaster"],
    16: ["Fridge-freezer (1)", "Fridge-freezer (2)", "Electric heater (1)", "Electric heater (2)", "Washing machine", "Dishwasher", "Computer site", "Television site", "Dehumidifier"],
    17: ["Freezer", "Fridge-freezer", "Tumble dryer", "Washing machine", "Computer site", "Television site", "Microwave", "Kettle", "TV site (bedroom)"],
    18: ["Fridge (garage)", "Freezer (garage)", "Fridge-freezer", "Washer dryer (garage)", "Washing machine", "Dishwasher", "Desktop computer", "Television site", "Microwave"],
    19: ["Fridge-freezer", "Washing machine", "Television site", "Microwave", "Kettle", "Toaster", "Bread-maker", "Games console", "Hi-fi"],
    20: ["Fridge", "Freezer", "Tumble dryer", "Washing machine", "Dishwasher", "Computer site", "Television site", "Microwave", "Kettle"],
    21: ["Fridge-freezer", "Tumble dryer", "Washing machine", "Dishwasher", "Food mixer", "Television", "Kettle", "Vivarium", "Pond pump"],
}


def name(house: int, channel: str) -> str:
    return NAMES[house][int(channel.replace("Appliance", "")) - 1]
