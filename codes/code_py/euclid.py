

def main():

    print('please two numbers')
    x = int(input('big number is :'))
    y = int(input('small number is :'))
    b = x%y
    while b!=0:
        x = y
        y = b
        b = x%y
    print(""y)
if __name__ == '__main__':
 main()
