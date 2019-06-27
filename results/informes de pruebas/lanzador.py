import informe_sgd_gpu_ram
import informe_sgld_cpu_ram
import informe_sgld_gpu_ram
import informe_sgld_multicore_ram

try:
    execfile('informe_sgld_cpu_ram.py')
    execfile('informe_sgld_multicore_ram.py')
    execfile('informe_sgd_gpu_ram.py')
    execfile('informe_sgld_gpu_ram.py')
except:
    pass


