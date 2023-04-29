def main():
    import pandas as pd
    import numpy as np
    import math

    age_group = 5

    filename = "src/data/population_data.csv"
    outputfile = "src/data/data.json"

    data = pd.read_csv(filename, dtype=str)
    data = data[
        [
            "ISO2_code",
            "Time",
            "AgeGrpStart",
            "Location",
            "PopMale",
            "PopFemale",
            "PopTotal",
        ]
    ]
    data["PopMale"] = list(
        map(
            lambda pop: int("{:.3f}".format(float(pop)).replace(".", "")),
            data["PopMale"],
        )
    )
    data["PopFemale"] = list(
        map(
            lambda pop: int("{:.3f}".format(float(pop)).replace(".", "")),
            data["PopFemale"],
        )
    )

    data["PopTotal"] = data["PopMale"] + data["PopFemale"]
    data = data.groupby(["Location"])

    oYears = []
    oPopType = """ 
    enum Locations {

     """
    oPop = """{
    """
    i = 20
    everyloc = True
    for location, rest in data:
        if str(rest["ISO2_code"].unique()[0]) != "nan":
            oPop += f'"{rest["ISO2_code"].unique()[0]}":'
        elif location[0] == "World":
            oPop += f'"W":'
        else:
            continue
        oPop += "{" + f'"name": "{location[0]}","info": ' + "{"
        print(location[0])
        oPopType += f'"{location[0]}",'
        year_data = rest.groupby(["Time"])

        for year, group in year_data:
            m = list(group["PopMale"])
            f = list(group["PopFemale"])
            if not str(year[0]) in oYears:
                oYears.append(str(year[0]))
            oPop += f"{year[0]}:"
            oPop += (
                "{"
                + f'"males": {list(np.add.reduceat(m, np.arange(0, len(m), age_group)))},"females": {list(np.add.reduceat(f, np.arange(0, len(f), age_group)))}'
                + "},"
            )
        oPop += "}" + "},"
        i -= 1
        if i == 0 and not everyloc:
            break
    oPop += "}"
    oPopType += "}"

    o = open(outputfile, "w")
    o.write(oPop)
    o.close()


if __name__ == "__main__":
    main()
