import os


def get_format_name(idx, lenght):
        '''
        :param idx: given img index, such as 1, 2, 3
        :lenght: format name length
        :return: return format img name like 000001, 000002, ...
        '''
        cnt = lenght - 1
        prefix = ''
        nmb = idx
        while idx // 10 != 0:
            cnt -= 1
            idx = idx // 10
        for i in range(cnt):
            prefix += '0'
        return prefix + str(nmb)


file_path = './00/image_0'
file_names = []

for i in range (0, 99):
    file_names.append('frame' + str(i) + '.jpg')

#print(file_names)

txt_path = "./00/times.txt"
with open(txt_path) as t:
    lines = t.readlines()

lines = [line.rstrip('\n') for line in lines]

print(lines)

i = 0
for name in file_names:
    src = os.path.join(file_path, name)
    dst = lines[i] + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
