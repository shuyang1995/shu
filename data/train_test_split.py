from sklearn.model_selection import train_test_split
import collections
def split(labels):
    train_f = open('train.txt', 'w+')
    test_f = open('test.txt', 'w+')
    table = collections.defaultdict(list)
    with open(labels, 'r') as f:
        for line in f:
            image_file, label = line.strip().split()
            table[label].append(image_file)
    for label in table.keys():
        l = table[label]
        n = len(l)
        #temp = [label]*n
        x_train, x_test = train_test_split(l, test_size = 0.1)
        for img_dir in x_train:
            train_f.write(img_dir +' '+label)
            train_f.write('\n')
        
        for img_dir in x_test:
            test_f.write(img_dir +' '+label)
            test_f.write('\n')
    train_f.close()
    test_f.close()

split('label.txt')
            
            
    
