import json

# with open("data/CISI.ALL.json") as file:
#     data = json.load(file)
#     result = []
#     for key, value in data.items():
#         if "title" in value and "text" in value:
#             item = {
#                 "id": str(int(value["id"])),
#                 "source": "cisi",
#                 "title": value["title"],
#                 "author": value["author"] if "author" in value else None,
#                 "content": value["text"],
#             }
#             result.append(item)
#     with open("data/data.json", mode="w") as ofile:
#         json.dump(result, ofile, indent=4)

with open("data/CRAN.QRY.json") as qfile:
    with open("data/CRAN.REL.json") as rfile:
        dqry = json.load(qfile)
        drel = json.load(rfile)
        drelarray = list(drel.values())
        result = []
        for key, value in dqry.items():
            if "text" in value and str(int(value["id"])) in drel:
                item = {
                    "id": str(int(value["id"])),
                    "source": "cran",
                    "text": value["text"],
                    "docs": list(set(drelarray[int(value["id"]) - 1].keys())),
                }
                item["docs"] = [int(x) for x in item["docs"]]
                item["docs"].sort()
                item["docs"] = [str(x) for x in item["docs"]]
                result.append(item)
        with open("data/data.json", mode="w") as ofile:
            json.dump(result, ofile, indent=4)

# with open("data/cisi_query.json") as file1:
#     with open("data/cran_query.json") as file2:
#         cisi = json.load(file1)
#         cran = json.load(file2)
#         result = []
#         for value in cisi:
#             value["source_id"] = value["id"]
#             value["id"] = f'cisi_{value["id"]}'
#             if "docs" in value:
#                 value["docs"] = [f"cisi_{item}" for item in value["docs"]]
#             result.append(value)
#         for value in cran:
#             value["source_id"] = value["id"]
#             value["id"] = f'cran_{value["id"]}'
#             if "docs" in value:
#                 value["docs"] = [f"cran_{item}" for item in value["docs"]]
#             result.append(value)
#         with open("data/data.json", mode="w") as ofile:
#             json.dump(result, ofile, indent=4)
