Para empezar el objeto mas importante es el mapa de convergencia $\kappa(\vec \theta)$, este hace el papel de la densidad efectiva de masa. Si bien la distribucion de masa es algo tridimensional $\rho(\vec r)$ lo que se hace es proyectarla a los ejes del plano de vision $\vec \theta = (\theta_1, \theta_2)$ de manera que la densidad es $\Sigma(\vec \theta)$. En particular se usa una constante de normalizacion la cual es $\Sigma_c$ lo que se refiere a la densidad critica.
$$
\kappa(\vec \theta) = \frac{\Sigma(\vec \theta)}{\Sigma_c}
$$
Donde el potencial bidimensional efectivo que controla la curva de los rayos de luz obedece la ecuación de Poisson, todo sera tomado en estas coordenadas del plano $\nabla_\theta \rightarrow \nabla$ y se usara a modo de notación simplificada
$$
\nabla^2  \psi (\vec \theta) = 2 \kappa (\vec \theta)
$$
De manera que construyendo la función de Green para el problema $\nabla^2 G(\vec \theta, \vec \theta') = \delta(\vec \theta - \vec \theta')$ lo cual tiene por solucion 
$$
G(\vec \theta, \vec \theta') = \frac{1}{2\pi} \ln |\vec{\theta} - \vec{\theta}'|
$$
De manera que se puede calcular el potencial a partir del mapa de convergencia usando la formula de Green
$$
\int G \nabla^2 \psi  - \psi \nabla^2 G \; d^2\theta' = \oint G \partial_n \psi - \psi \partial_n G \; d \ell
$$
El lado derecho puede ser definido a partir de distintas condiciones, la primera es que la funcion de Green decae mientras nos alejamos, por tanto evaluada en los bordes tiende a 0. El potencial bidimensional por otro lado no decae pues la distribución de materia oscura permea todo el universo observable, y la función de Green queda libre a definirse su derivada en el borde para determinarla por completo, de manera que el lado derecho puede describirse como una constante a elegir libremente:
$$
\oint G \partial_n \psi - \psi \partial_n G \; d \ell = 0 - \oint \psi \partial_n G \; d \ell = -\psi_0
$$
Puede ser elegido como 0 si se desea, el potencial solo importan sus derivadas.
El lado izquierdo por otro lado puede ser resuelto utilizando $\nabla^2  \psi (\vec \theta) = 2 \kappa (\vec \theta)$,  $\nabla^2 G(\vec \theta, \vec \theta') = \delta(\vec \theta - \vec \theta')$ de manera que:
$$
\int G \nabla^2 \psi  - \psi \nabla^2 G \; d^2\theta' = \int G \,2 \kappa - \psi \, \delta(\vec \theta - \vec \theta') d^2 \theta'
$$
$$
 \int 2 G \kappa d^2 \theta'  - \psi(\vec \theta)= -\psi_0
$$
Despejando se obtiene el potencial bidimensional
$$
\psi(\vec{\theta})  =\frac{1}{\pi} \int \kappa(\vec{\theta}') \ln |\vec{\theta} - \vec{\theta}'| d^2\theta'+ \psi_0
$$

# Curvatura de rayos
El angulo de curvatura viene a ser la gradiente del potencial, la gradiente puede ingresar a la integral y obtener una formula mas resumida
$$
\vec{\alpha}(\vec{\theta}) = \nabla \psi = \frac{1}{\pi} \int \kappa(\vec{\theta}') \frac{\vec{\theta} - \vec{\theta}'}{|\vec{\theta} - \vec{\theta}'|^2} d^2\theta',
$$
Ademas es posible linearizar a modo de obtener una matriz que describe las transformaciones
$$
\mathbb{A}_{ij} \equiv \frac{\partial \vec{\beta}}{\partial \vec{\theta}} = \left( \delta_{ij} - \frac{\partial \alpha_i(\vec{\theta})}{\partial \theta_j} \right) = \left( \delta_{ij} - \frac{\partial^2 \psi(\vec{\theta})}{\partial \theta_i \partial \theta_j} \right) = \mathbb{M}^{-1}.
$$
Se utiliza una notacion resumida
$$
\frac{\partial^2 \psi}{\partial \theta_i \partial \theta_j} \equiv \psi_{ij}.
$$
De manera que cantidades pueden ser definidas por este tensor $\psi_{ij}$ 
$$
\kappa = \frac{1}{2} (\psi_{11} + \psi_{22}) = \frac{1}{2} \operatorname{tr} \psi_{ij}.
$$
Asi como definir el "shear", el shear simetrico
$$
\gamma_1(\vec{\theta}) = \frac{1}{2} (\psi_{11} - \psi_{22}) \equiv \gamma(\vec{\theta}) \cos \left[ 2\phi(\vec{\theta}) \right],
$$
como el componente antisimetrico
$$
\gamma_2(\vec{\theta}) = \psi_{12} = \psi_{21} \equiv \gamma(\vec{\theta}) \sin \left[ 2\phi(\vec{\theta}) \right].
$$
Lo cual permite escribir de manera resumida
$$
\mathbb{A} = \begin{pmatrix} 1 - \kappa - \gamma_1 & -\gamma_2 \\ -\gamma_2 & 1 - \kappa + \gamma_1 \end{pmatrix} = (1 - \kappa) \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} - \gamma \begin{pmatrix} \cos 2\phi & \sin 2\phi \\ \sin 2\phi & -\cos 2\phi \end{pmatrix}
$$
El mapa de convergencia afecta la magnificacion, la parte del Shear introduce astigmatismo lo cual modifica los circulos a elipses,

## Lista de ejemplos

![[Pasted image 20240803210344.png]]