def main():
    import pandas as pd
    import numpy as np
    import math

    age_group = 3

    filename = "src/data/population_age.csv"
    outputfile = "src/data/data.ts"

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
    oPop = """
    import type { Data } from "@/types";

    export default {
    """
    i = 10
    everyloc = False
    for location, rest in data:
        print(location[0])
        oPop += (
            f'"{location[0]}":'
            + "{"
            + f"'code': \"{rest['ISO2_code'].unique()[0] if rest['ISO2_code'].unique()[0] != 'nan' else 'none'}\",'info': " + "{"
        )
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
                + f"'males': {list(np.add.reduceat(m, np.arange(0, len(m), age_group)))},'females': {list(np.add.reduceat(f, np.arange(0, len(f), age_group)))}"
                + "},"
            )
        oPop += "}"+"},"
        i -= 1
        if i == 0 and not everyloc:
            break
    oPop += "} as Data"
    oPopType += "}"

    o = open(outputfile, "w")
    o.write(oPop)
    o.close()


if __name__ == "__main__":
    main()
