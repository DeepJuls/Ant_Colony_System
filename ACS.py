#Algoritmo para implementar la metaheurística constructiva Sistema de Colonias de Hormigas (ACS).
#Julián García-Abadillo Velasco

import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen

class ACS():

	def __init__(self,cost_array,task_list,oper_list,
					beta,num_ant,generations,monitor,
					initial_pheromone,ro,phi,penalty,
					alfa,q0):

		self.cost_array = cost_array
		self.task_list = task_list
		self.oper_list = oper_list

		self.num_task = len(self.task_list)
		self.num_oper = len(self.oper_list)

		#Vector donde el elemento i representa al i-ésimo operario realizando la primera tarea
		self.initial_array = [initial_pheromone]*self.num_oper
		#Tensor donde el elemento ijk representa el arco donde la tarea i-ésima es realizada por 
		#el operario j-ésimo y la tarea iésima + 1 es realizada por el operario k-ésimo.
		self.transition_matrix = []
		for i in range(len(task_list)-1):
			self.transition_matrix.append([])
			for j in range(len(oper_list)):
				self.transition_matrix[i].append([])
				for k in range(len(oper_list)):
					self.transition_matrix[i][j].append(initial_pheromone)

		self.best_combination = None
		self.best_fitness = float('inf')

		self.alfa = alfa
		self.beta = beta
		self.num_ant = num_ant
		self.ro = ro
		self.phi = phi
		self.q0 = q0
		self.generations = generations
		self.penalty = penalty
		self.monitor = monitor

		self.Main()

	#Función de evaluación: 
	#Calcula el coste y penaliza las soluciones no válidas.
	def Fitness(self,chromosome):
		fit = 0
		free_time = self.oper_list[:]
		for i in range(len(chromosome)):
			fit += self.cost_array[chromosome[i]][i]
			free_time[chromosome[i]] -= self.task_list[i]
		for item in free_time:
			if item < 0:
				fit *= self.penalty
		return fit

	#Función ruleta: 
	#Esta función escoge probabilísticamente un elemento
	#dada una función de distribución acumulada.
	def Roulette(self,distribution):
		u = np.random.uniform()
		for i in range(len(distribution)):
			if distribution[i] >= u:
				return i

	#Distribución inicial:
	#Esta función devuelve el nodo en el que empiza una hormiga
	#siguiendo la distribución de probabilidad acumulada. 
	#Dependen tanto de la feromona como de la heurística.
	def Initial_distribution(self):
		density = []
		for i in range(self.num_oper):
			density.append(self.initial_array[i]*(1/self.cost_array[i][0])**self.beta)
		total = sum(density)
		distribution = []
		for i in range(len(density)):
			distribution.append(sum(density[:i+1])/total)
		return self.Roulette(distribution)

	#Función de transición AS:
	#Esta función devuelve el nodo a seguir utilizando
	#la regla del algoritmo AS simple. En este caso se aplica
	#cuando el valor de q es superior a q0.
	def AS_Transition(self,current,task):
		density = []
		for i in range(self.num_oper):
			a = self.transition_matrix[task][current][i]
			b = (1/self.cost_array[i][task])**self.beta
			density.append(a*b)
		total = sum(density)
		distribution = []
		for i in range(len(density)):
			distribution.append(sum(density[:i+1])/total)
		return self.Roulette(distribution)

	#Función de transición ACS:
	#Función de transición específica del algoritmo ACS.
	#Se escogerá la mejor solución con probabilidad q0
	#o la solución de AS_Transition() con probabilidad 1-q0.
	def ACS_Transition(self,current,task):
		q = np.random.uniform()
		if q <= self.q0:
			best = [None, 0]
			for i in range(self.num_oper):
				a = self.transition_matrix[task][current][i]**self.alfa
				b = (1/self.cost_array[i][task])**self.beta
				if a*b > best[1]:
					best = [i,a*b]
			return i
		else:
			return self.AS_Transition(current,task)

	#Función de actualización local de feromona:
	#Esta actualización ocurre en cada arco visitado por una hormiga.
	def Local_Actualization(self,arch):
		#Actualización de la feromona en los trayectos punto inicial - primera tarea.
		if len(arch) == 1:
			previous = self.initial_array[arch[0]]
			new = (1-self.phi)*previous + self.phi/(self.num_ant*self.cost_array[arch[0]][0])
			self.initial_array[arch[0]] = new
		#Actualización del tensor de feromonas, que contiene el resto de arcos.
		else:
			previous = self.transition_matrix[arch[0]][arch[1]][arch[2]]
			new = (1-self.phi)*previous + self.phi/(self.num_ant*self.cost_array[arch[2]][arch[0]+1])
			self.transition_matrix[arch[0]][arch[1]][arch[2]] = new
		return

	#Función de actualización global de feromona:
	#Esta actualización ocurre solo en el camino trazado por la mejor hormiga.
	def Global_Actualization(self,index):
		#Actualización de la feromona en los trayectos punto inicial - primera tarea.
		previous = self.initial_array[self.best_combination[index][0]]
		new = (1-self.ro)*previous + self.ro/self.best_fitness
		self.initial_array[self.best_combination[index][0]] = new
		#Actualización del tensor de feromonas, que contiene el resto de arcos.
		for i in range(len(self.best_combination[index])-1):
			previous = self.transition_matrix[i][self.best_combination[index][i]][self.best_combination[index][i+1]]
			new = (1-self.ro)*previous + self.ro/self.best_fitness
			self.transition_matrix[i][self.best_combination[index][i]][self.best_combination[index][i+1]] = new
		return

	#Esta función grafica la feromona depositada en cada nodo en una determinada iteración.
	#Es específica para el problema estudiado, no funcionaría para otros.
	#Se encuentra desactivada por defecto (comentada en la función Main()).
	def Paint_ants(self,iteration):
		def drawArrow(A, B,alpha):
			plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],  length_includes_head=True,width=2*alpha+0.7,color="gray",alpha=alpha**2+0.4)
			return

		xl = [10,40,70,100]
		yl = [100,80,60,40,20]
		plt.figure(1)
		plt.xlim(left=-25,right=120)
		plt.xticks([])
		plt.yticks([])
		plt.scatter(-15,60,s=1000,c='red')
		plt.annotate("0", (-17,57.5),size=20,color="white")
		for i in range(len(xl)):
			for j in range(len(yl)):
				plt.scatter(xl[i],yl[j],s=1000,c="b")
				plt.annotate(str(j+1), (xl[i]-1.8,yl[j]-2.1),size=20,color="white")
		for i in range(len(self.initial_array)):
			drawArrow([-15,60],[10,yl[i]],alpha=self.initial_array[i])
		for i in range(len(self.transition_matrix)):
			for j in range(len(self.transition_matrix[i])):
				for k in range(len(self.transition_matrix[i][j])):
					drawArrow([xl[i],yl[j]],[xl[i+1],yl[k]],alpha=self.transition_matrix[i][j][k])
		Popen(["rm","-f","-r","gif"])
		Popen(["mkdir","gif"])
		plt.savefig("gif/"+str(iteration)+".png")
		plt.close()

	#Función principal que sigue el pseudocódigo de las páginas 21 y 22 de los apuntes.
	def Main(self):
		#Inicialización de vectores para la monitorización global del proceso a través de los fitness.
		best_array = []
		mean_array = []
		stdv_array = []
		for gen in range(self.generations+1):
			ants = []
			#Primero se añade un primer nodo a cada hormiga
			for ant in range(self.num_ant):
				ants.append([self.Initial_distribution()])
				self.Local_Actualization(ants[-1])
			#A continuación se van añadiendo el resto de nodos.
			for i in range(len(self.task_list)-1):
				for ant in ants:
					ant.append(self.ACS_Transition(ant[-1],i))
					self.Local_Actualization([i]+ant[-2:])
			#Todas las hormigas están completas y se pasa a evaluarlas.
			#También se almacenan todos los fitness para la monitorización.
			gen_fitness = []
			for ant in ants:
				candidate_fitness = self.Fitness(ant)
				gen_fitness.append(candidate_fitness)
				if candidate_fitness < self.best_fitness:
					self.best_fitness = candidate_fitness
					self.best_combination = [ant]
				elif candidate_fitness == self.best_fitness and ant not in self.best_combination:
					self.best_combination.append(ant)
			#Se calcula la media, varianza y mejor fitness de todos los de cada generación.
			best_array.append(min(gen_fitness))
			mean_array.append(np.mean(gen_fitness))
			stdv_array.append(np.std(gen_fitness))
			#Una vez actualizada la(s) mejor(es) hormiga(s), se realiza la actualización global con ella(s).
			for ant in range(len(self.best_combination)):
				self.Global_Actualization(ant)
			if gen % self.monitor == 0:
				#self.Paint_ants(gen)

				print("Generación {}/{}:".format(gen,self.generations))
				print("······························································································")
				print("Mejor coste: {}".format(self.best_fitness))
				print("Mejor(es) combinacion(es): {}\n{}".format(len(self.best_combination),self.best_combination))
				print("······························································································\n\n")
		#Graficar la evolución de la media, desviación y mejor resultado para el fitness en cada generación.
		x_axis = list(range(self.generations))
		fig, (ax1, ax2, ax3) = plt.subplots(3)
		ax1.plot(x_axis, mean_array[:-1],"r",alpha=0.8,linewidth=0.5)
		ax1.set_title("Media")
		ax1.axes.get_xaxis().set_visible(False)
		ax2.plot(x_axis, stdv_array[:-1],"b",alpha=0.8,linewidth=0.5)
		ax2.set_title("Desviación típica")
		ax2.axes.get_xaxis().set_visible(False)
		ax3.plot(x_axis, best_array[:-1],"g",alpha=0.8,linewidth=0.5)
		ax3.set_title("Mejor")
		plt.show()
		
#Implementación del problema en concreto
cost_array = [[2,3,4,1],[3,2,3,2],[2,2,1,2],[3,3,3,3],[2,1,2,2]]
task_list = [3,2,2,3]
oper_list = [4,5,3,4,4]
#Con esta semilla encuentra 5 soluciones en unos 3000 ciclos.
np.random.seed(4)
ACS(
#Parámetros del problema
cost_array = cost_array,
task_list = task_list,
oper_list = oper_list,
penalty = 5, #Penalización por no cumplir alguna condición.
#Parámetros del algoritmo
generations = 5000, #Número máximo de generaciones.
num_ant = 50, #Número de hormigas
alfa = 1, #Parámetro alfa (importancia de la feromona)
beta = 0, #Parámetro beta (importancia de la información heurística)
ro = 0.1, #Parámetro ro (evaporación global)
phi = 0.1, #Parámetro phi (evaporación local)
q0 = 0.9, #Probabilidad de escoger el mejor camino en la transición.
initial_pheromone = 1, #Cantidad de feromomona al principio
#Parámetros de monitorización.
monitor = 500) #Cada cuantas generaciones se sacan imágenes y se muestra la información en pantalla.

