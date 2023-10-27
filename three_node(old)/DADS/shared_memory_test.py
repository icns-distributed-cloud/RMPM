from multiprocessing import Process, Value, Array

def f(n, a, num):
    n.value = num
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p1 = Process(target=f, args=(num, arr, 1))
    p1.start()
    p1.join()

    # p2 = Process(target=f, args=(num, arr, 2))
    # p2.start()
    # p2.join()

    print(num.value)
    print(arr[:])