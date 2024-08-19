import pandas as pd

# The file has no names for columns.
columns = ["buying","maint","doors","persons","lug_boot","safety","class"]
df = pd.read_csv('car.data', names=columns)

replacement_dict = {
    "buying": {"vhigh": 3, "high": 2, "med": 1, "low": 0},
    "maint": {"vhigh": 3, "high": 2, "med": 1, "low": 0},
    "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons": {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety": {"low": 0, "med": 1, "high": 2},
    "class": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
}

for column, mapping in replacement_dict.items():
    df[column] = df[column].replace(mapping)

print(df)

df_x = df[["buying","maint","doors","persons","lug_boot","safety"]]
df_y = df[["class"]]

