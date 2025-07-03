import pyvista as pv
import numpy as np

# 1. Carica i dati delle due mesh
print("Caricamento mesh...")
reference_mesh = pv.read('/home/luca/HPC/PDE/nmpde-projects/golden_reference/build/solution_9999.pvtu')
adaptive_mesh = pv.read('/home/luca/HPC/PDE/nmpde-projects/build/output_9999.vtu')
print("Caricamento completato.")


# 2. RISOLUZIONE BUG 1: Rinomina il campo dati sulla reference mesh
#    per evitare la collisione di nomi durante l'interpolazione.
reference_mesh.rename_array('u', 'u_reference')


# 3. Interpola i dati
print("Inizio interpolazione...")
# Ora interpoliamo. La adaptive_mesh conterrà sia il suo campo 'u' originale,
# sia il nuovo campo 'u_reference' interpolato.
interpolated_mesh = adaptive_mesh.interpolate(reference_mesh, sharpness=2)
print("Interpolazione completata.")


# 4. Estrai i due vettori di dati DISTINTI
u_adaptive = interpolated_mesh['u']
u_reference_interpolated = interpolated_mesh['u_reference'] # Usiamo il nuovo nome


# 5. SANITY CHECK
print(f"Norma della soluzione adattiva: {np.linalg.norm(u_adaptive)}")
print(f"Norma della soluzione di riferimento interpolata: {np.linalg.norm(u_reference_interpolated)}")


# 6. Calcola l'errore
error_vector = u_adaptive - u_reference_interpolated
# Ora questo dovrebbe essere un valore piccolo, ma non zero!
print(f"\nErrore L2 (norma vettoriale): {np.linalg.norm(error_vector)}")


# 7. Calcola la vera norma L2
interpolated_mesh['error_squared'] = error_vector**2

# Integriamo. La funzione restituisce un oggetto PyVista Table.
integration_result_table = interpolated_mesh.integrate_data('error_squared')

# --- PASSO DI DEBUG: STAMPIAMO L'OGGETTO PER VEDERE COSA CONTIENE ---
print("\n--- Contenuto della tabella di integrazione (DEBUG) ---")
print(integration_result_table)
print("-----------------------------------------------------\n")

# Estrai il valore numerico usando il nome corretto che vedi stampato sopra.
# Molto probabilmente, PyVista ha chiamato l'array come il campo che hai
# integrato, cioè 'error_squared'.
try:
    # Prova con il nome più probabile
    correct_array_name = 'error_squared'
    integral_of_error_squared = integration_result_table[correct_array_name][0]
except KeyError:
    # Se non funziona, usa il primo array che trovi (spesso l'unico)
    correct_array_name = integration_result_table.array_names[0]
    integral_of_error_squared = integration_result_table[correct_array_name][0]


# Ora possiamo calcolare la radice quadrata del numero
true_l2_error = np.sqrt(integral_of_error_squared)

print(f"Errore L2 (integrato sul volume): {true_l2_error}")