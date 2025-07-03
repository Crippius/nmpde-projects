import pyvista as pv
import numpy as np

# 1. Carica i dati delle due mesh
print("Caricamento mesh...")
reference_mesh = pv.read('path/to/your/reference_solution.vtu')
adaptive_mesh = pv.read('path/to/your/adaptive_solution.vtu')
print("Caricamento completato.")

# 2. Estrai e COPIA la soluzione adattiva ORIGINALE
#    Il .copy() è fondamentale per assicurarsi di non modificare
#    questo vettore accidentalmente in seguito.
u_adaptive = adaptive_mesh['u'].copy()

# 3. Interpola la soluzione di riferimento sulla griglia adattiva
print("Inizio interpolazione (può richiedere tempo)...")
# La funzione .interpolate() aggiunge i campi interpolati alla adaptive_mesh
adaptive_mesh.interpolate(reference_mesh, sharpness=2, inplace=True)
print("Interpolazione completata.")

# Ora 'adaptive_mesh' contiene DUE campi 'u': quello originale e quello
# interpolato (il nome potrebbe variare, controlla con print(adaptive_mesh)).
# Per essere sicuri, estraiamo il campo interpolato.
# PyVista spesso aggiunge un suffisso, ma assumiamo che sovrascriva 'u'.
u_reference_interpolated = adaptive_mesh['u']

# 4. SANITY CHECK (Controllo di sicurezza)
#    Stampa le norme dei singoli vettori. Se queste non sono zero
#    ma l'errore finale è zero, significa che i due vettori sono identici.
print(f"Norma della soluzione adattiva: {np.linalg.norm(u_adaptive)}")
print(f"Norma della soluzione di riferimento interpolata: {np.linalg.norm(u_reference_interpolated)}")

# 5. Calcola l'errore
error_vector = u_adaptive - u_reference_interpolated
l2_error = np.linalg.norm(error_vector)

print(f"\nErrore L2 calcolato: {l2_error}")


# Aggiungi il vettore errore alla mesh adattiva come un nuovo campo
adaptive_mesh['error_squared'] = error_vector**2

# Usa la funzione di PyVista per integrare questo campo sull'intero volume della mesh
# Questo calcola ∫(errore²) dV
integral_of_error_squared = adaptive_mesh.integrate_data('error_squared')

# La norma L2 è la radice quadrata di questo integrale
true_l2_error = np.sqrt(integral_of_error_squared)

print(f"Errore in norma L2 (integrato): {true_l2_error}")