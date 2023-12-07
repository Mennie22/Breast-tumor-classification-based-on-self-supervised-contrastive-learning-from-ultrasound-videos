import os
from pretrain.main import args

def get_indices():
    indices = {}
    person = 0
    mid = []
    filelists = os.listdir(args.pretrain_data_dir)
    sortf = []

    for f in filelists:
        sortf.append(
            int(f.split('_')[0] + '00000') + int(f.split('_')[1] + '000') + int(os.path.splitext(f.split('_')[2])[0]))
        sortf.sort()

    sort_file = []
    for num in sortf:
        for f in filelists:
            if num == int(f.split('_')[0] + '00000') + int(f.split('_')[1] + '000') + int(
                    os.path.splitext(f.split('_')[2])[0]):
                sort_file.append(f)

    Max = int(os.path.splitext(sort_file[-1])[0].split('_')[0])  # 得到人的数目

    num = 1
    indices = {0: {}}

    pics = 0

    for i, f in enumerate(sort_file):
        mid = []
        ends = str(os.path.splitext(f)[0])
        for j in range(len(ends)):
            if ends[j] == '_':
                mid.append(j)

        if person == int(ends[:mid[0]]):

            if num == int(ends[mid[0] + 1:mid[1]]):
                pics += 1

            else:
                indices[person][num] = pics

                num += 1
                pics = 1
        else:
            indices[person][num] = pics

            num = 1
            person += 1
            pics = 1
            indices[person] = {num: {}}

    indices[person][num] = pics
    print("Successfully get indices")
    return indices
