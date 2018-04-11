def Robbins_monro(function,grad,number_variable_gradient):

    print('Start Robbins monro algorithm \n')

    init_learning_rate = 1.5
    print('Initial value of learning rate is %s \n'%init_learning_rate)

    stepsize = 1000
    print('Stepsize is %s \n'%stepsize)
    print('Number of traning data is 100')

    print('Make initial value of estimation \n')
    init_value = np.array([range(-1000,1020,20) for i in range(number_variable_gradient)])
    print('initial value = \n %s'%init_value)
    print('Complicated, start to estimate oprimal solution\n')

    possibility_solution = np.zeros((number_variable_gradient,101))
    for n in range(len(init_value[0,:])):
        old_estimation = init_value[:,n]
        for step in np.arange(1,stepsize+1):
            n_step = init_learning_rate/step
            old_estimation_t = old_estimation

            new_estimation = old_estimation - n_step*grad(old_estimation)
            old_estimation = new_estimation

        if n % 10 == 0:
            print('%s taimes step is comlicated \n'%n)
        possibility_solution[:,n] = new_estimation
    print('Resolution is complicated \n Display option of solution \n E vry columns is option of solution\n' )
    print(possibility_solution)
    estimation = possibility_solution[:,0]
    for j in range(len(possibility_solution[:,0])):
        if function(estimation) > function(possibility_solution[:,j]):
            estimation = possibility_solution[:,j]


    print('Estimation is comlicated \n Most suitablest solutin is \n %s'%estimation )
    print('Functin vale is %s'%function(estimation))



def f(x):
    return x**2 + 4
def g(x):
    return 2*x

Robbins_monro(f,g,1)
