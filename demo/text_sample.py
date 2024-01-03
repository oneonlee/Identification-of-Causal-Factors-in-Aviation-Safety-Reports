
if __name__ == "__main__":
    report_texts = []
    for i in range(1,5):
        file = open(f'./samples/sample{i}.txt', 'r',encoding='UTF-8')
        text=file.read()
        report_texts.append(text)
        print(text)
        file.close()