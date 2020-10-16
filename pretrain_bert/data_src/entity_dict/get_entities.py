import json
import copy


def alldata2list(train_data_path):
    """将所有训练数据合并到一个列表
    :return: train_data (List[Dict])
    """
    train_data = []
    for idx in range(1, 401):
        with open(train_data_path / f'train_V2_{idx}.json', encoding='gbk') as f:
            data = json.load(f)
            train_data.append(data)
    return train_data


def construction(train_data):
    """构造sentence.txt和与其对应的tags.txt
    @:param train_data (List[Dict])
    """
    with open('./dict.txt', 'w', encoding='utf-8') as f:
        for data in train_data:
            # 取文本和标注
            # 去掉原文本前后的回车换行符
            # 将原文本中间的回车换行符替换成r（符合源数据标注规则）
            # 将特殊字符替换为UNK
            data_ori = list(data['originalText'].strip().replace('\r\n', '✄').replace(' ', '✄'))
            data_text = copy.deepcopy(data_ori)
            data_entities = data['entities']
            for entity in data_entities:
                start_ind = entity['start_pos'] - 1
                end_ind = entity['end_pos'] - 1
                # 获取实体
                en = data_text[start_ind: end_ind + 1]
                if len(en) > 1:
                    f.write(''.join(en) + '\n')


if __name__ == '__main__':
    from pathlib import Path

    train_data_path = Path('./train')
    train_data = alldata2list(train_data_path)
    construction(train_data)
