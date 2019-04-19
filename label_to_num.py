import sys

prop_train_path = 'prop_train.txt'
svm_train_path = 'svm_train.txt'

def label_to_num(label_name):
    return {
        'background': 0,
        'Great_Pyrenees': 1,
        'wire-haired_fox_terrier': 2,
        'Irish_setter': 3
    }.get(label_name, -1)


if __name__ == '__main__':
    paths = [prop_train_path, svm_train_path]
    
    for path in paths:
        _filew = open(path.split('.')[0]+'_convert.txt', 'w')
        with open(path, 'r') as _file:
            lines = _file.read().splitlines()
            for line in lines:
                comps = line.split(' ')
                label_num = label_to_num(comps[1])
                if label_num == -1:
                    print("some category not known , " + comps[1])
                    sys.exit(-1)
                _filew.write(comps[0] + " " + str(label_num) + "\n")
        _filew.close()
