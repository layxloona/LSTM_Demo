def load_data(file_path, flap=30):
    with open(file_path) as f:
        text = f.read()

    datasets = []
    text = text.split("\n")
    for i in text:
        temp = i.split(" ")
        temp1 = []
        state = True
        for j in range(len(temp)):
            try:
                temp1.append(float(temp[j]))
            except:
                pass

        if len(temp1) == 36:
            temp1.pop(28)
            temp1.pop(28)
            temp1.pop(30)
            temp1.pop(30)
            for t in temp1:
                if t == 0.:
                    state = False
            if state:
                datasets.append(temp1)

    # Build features: position (32), velocity (32), acceleration (32) => 96-d
    x_data = []
    y_data = []
    # Need i >= 2 to compute acceleration, and i+flap within bounds
    last_index_for_input = len(datasets) - 1 - flap
    for i in range(2, last_index_for_input):
        pos = datasets[i]
        prev = datasets[i - 1]
        prev2 = datasets[i - 2]
        vel = [p - q for p, q in zip(pos, prev)]
        acc = [p - 2 * q + r for p, q, r in zip(pos, prev, prev2)]
        x_data.append(pos + vel + acc)
        y_data.append(datasets[i + flap])

    return x_data, y_data