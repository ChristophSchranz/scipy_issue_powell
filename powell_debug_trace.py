import numpy as np
import scipy.optimize as spo
import scipy.stats as sps


#####################################################
print('\n\nStandard approach, works not')
x, y = np.loadtxt('xy.csv', delimiter=',').T

def calc_sse(c_s, x, y):
    print(c_s)
    errors = y - (c_s[0] + x * c_s[1])
    return np.sum(errors ** 2)

ls_res = sps.linregress(x, y)
print(f'LS inter: {ls_res.intercept}')
print(f'LS slope: {ls_res.slope}')
ls_sse = calc_sse((ls_res.intercept, ls_res.slope), x, y)
print(f'LS SSE error: {ls_sse}')

start = [2.25, 0.47]

print('Powell minimization with scipy.fmin_powell:')
print(spo.fmin_powell(calc_sse, start, args=(x, y)))

#####################################################
# fmin_powell with scaling of the parameter space to relatively decrease the step_width, works
print("\n\nfmin_powell with scaling of the parameter space to relatively decrease the step_width, works")
scale = 0.1
x, y = np.loadtxt('xy.csv', delimiter=',').T

def calc_sse(c_s, x, y):
    c_s = [scale*v for v in c_s]
    errors = y - (c_s[0] + x * c_s[1])
    print(c_s)
    return np.sum(errors ** 2)

ls_res = sps.linregress(x, y)
print(f'LS inter: {ls_res.intercept}')
print(f'LS slope: {ls_res.slope}')
ls_sse = calc_sse((ls_res.intercept, ls_res.slope), x, y)
print(f'LS SSE error: {ls_sse}')

start = [2.25, 0.47]
c_s = [scale*v for v in start]

print('Powell minimization:')
print(spo.fmin_powell(calc_sse, start, args=(x, y)))


#####################################################
# fmin_powell with direction matrix, works
print("\n\nfmin_powell with direction matrix, works")
x, y = np.loadtxt('xy.csv', delimiter=',').T

def calc_sse(c_s, x, y):
    errors = y - (c_s[0] + x * c_s[1])
    print(c_s)
    return np.sum(errors ** 2)

ls_res = sps.linregress(x, y)
print(f'LS inter: {ls_res.intercept}')
print(f'LS slope: {ls_res.slope}')
ls_sse = calc_sse((ls_res.intercept, ls_res.slope), x, y)
print(f'LS SSE error: {ls_sse}')

start = [2.25, 0.47]

print('Powell minimization:')
print(spo.fmin_powell(calc_sse, start, args=(x, y), direc=[[1,-0.5],[-0.5,1]]))


#####################################################
# fmin_powell standard on a horizontal valley, works
print('\n\nfmin_powell standard on a horizontal valley, works')
x, y = np.loadtxt('xy.csv', delimiter=',').T
phi = 0.0795
rotation_m = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
x = np.matmul(np.concatenate([x, y]).reshape(-1,2), rotation_m)[:,0]
y = np.matmul(np.concatenate([x, y]).reshape(-1,2), rotation_m)[:,1]

def calc_sse(c_s, x, y):
    # c_s = [scale*v for v in c_s]
    errors = y - (c_s[0] + x * c_s[1])
    print(c_s)
    return np.sum(errors ** 2)

ls_res = sps.linregress(x, y)
print(f'LS inter: {ls_res.intercept}')
print(f'LS slope: {ls_res.slope}')

ls_res = sps.linregress(x, y)
print(f'LS inter: {ls_res.intercept}')
print(f'LS slope: {ls_res.slope}')
ls_sse = calc_sse((ls_res.intercept, ls_res.slope), x, y)
print(f'LS SSE error: {ls_sse}')

start = [2.25, 0.47]
start = list(np.matmul(np.array(start).reshape(-1,2), rotation_m))

# print("\nPowell minimization with pdfo's COBYLA:")
# print(pdfo(calc_sse, start, args=(x, y), method='COBYLA'))

print('Powell minimization:')
print(spo.fmin_powell(calc_sse, start, args=(x, y), direc=[[1,-0.5],[-0.5,1]]))
