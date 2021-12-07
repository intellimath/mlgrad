# coding: utf-8

from mlgrad.func cimport Func

cdef class Average_SAG(Average):
    #
    def __init__(self, Func func, tol=1.0e-5, h=1.0, n_iter=1000, ls=0):
        """
        """
        self.func = func
        self.tol = tol
        self.n_iter = n_iter
        self.h = h
        self.ls = ls
    #
    cpdef evaluate(self, float_1d Y):
        self.fit(Y)
        return self.y
    #
    cdef fit_init(self, float_1d Y, object y=None, object sigma=None):
        cdef int N = Y.shape[0]
        
        Average.fit_init(self, Y, y, sigma)
        
        init_rand()
                
        self.G_all = np.zeros((N,), 'd')
        self.G_sum = 0.0
        
        #self.y_mean = 0
        #self.y2_mean = 0
        self.h_all = np.zeros((N,), 'd')        
        for k in range(N):
            self.h_all[k] = self.h
        self.h_mean = self.h
        
        self.fill_tables_u(Y)        
    #
    cdef fill_tables(self, float_1d Y):
        cdef int N = len(Y)
        cdef int k
        cdef float g, y
        cdef float Nd = N
        
        for k in range(N):
            y = Y[k]
            
            # значение штрафной функции
            g = -self.func.derivative(y - self.y)
            # добавляем в сумму производных
            self.G_sum += g
            # сохраняем в таблице производных
            self.G_all[k] = g
    #    
    cdef object fit_epoch_u(self, float_1d Y):
        cdef bint stop
        cdef int N = len(Y)
        cdef int j, k
        cdef float Nd = N
        cdef float yk, v, y_prev
        cdef float g
        cdef float h, h_min
        cdef float G_mean
        cdef float y_sum, y2_sum
        
        self.y_prev = self.y
        
        #y_sum = 0
        #y2_sum = 0
                
        for j in range(N):
            # очередной случайный индекс
            k = rand(N)
            yk = Y[k]
                        
            # значение функции потерь
            g = -self.func.derivative(yk - self.y)
            
            # обновляем параметры модели,
            # таблицу с производными и среднее значение производных
            self.G_sum += (g - self.G_all[k])
            self.G_all[k] = g            
            G_mean = self.G_sum / Nd

            if self.ls:
                g2 = G_mean * g

                if g2 > 1.0e-8:
                    h = self.line_search(yk, G_mean, g2, self.h_mean)
                else:
                    h = self.h_mean

                self.h_mean = Nd / (Nd/self.h_mean + (1.0/h - 1.0/self.h_all[k]))
                self.h_all[k] = h 
            else:
                h = self.h

            self.y -= h * G_mean
            
            #y_sum += self.y
            #y2_sum += self.y * self.y
        
        #self.y_mean = y_sum / Nd
        #self.y2_mean += y2_sum / Nd
            
    #
    cdef float line_search(self, float xk, float G, float G2, float h):
        cdef int j
        cdef float lval0, lval
        cdef float a=0.5, b=0.24
                            
        lval0 = self.func.evaluate(xk - self.y)
        self.y -= h * G

        for j in range(64):
    
            lval = self.func.evaluate(xk - self.y)
            if lval <= lval0 - b*h*G2:
                self.y += h * G
                return h

            self.y += (1.0 - a) * h * G
    
            h *= a

        return h
    #
