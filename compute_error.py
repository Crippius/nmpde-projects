import os
# Imposta la modalità headless PRIMA di importare qualsiasi altra cosa.
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import sys
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

def main():
    """
    Calcola l'errore L2 tra una soluzione di test e una di riferimento,
    usando un KD-Tree per un'interpolazione robusta e performante.
    """
    parser = argparse.ArgumentParser(description="Calcola l'errore L2 e lo salva su file.")
    parser.add_argument("--reference", required=True, type=Path, help="Percorso al file della mesh di riferimento.")
    parser.add_argument("--test-solution", required=True, type=Path, help="Percorso al file della soluzione di test.")
    parser.add_argument("--output-file", required=True, type=Path, help="Percorso del file dove scrivere il risultato numerico.")
    args = parser.parse_args()

    print(f"Info: Avvio calcolo errore...", file=sys.stderr)
    
    if not args.reference.exists():
        print(f"Errore: File di riferimento non trovato in '{args.reference}'", file=sys.stderr)
        sys.exit(1)
    if not args.test_solution.exists():
        print(f"Errore: File di test non trovato in '{args.test_solution}'", file=sys.stderr)
        sys.exit(1)

    try:
        #pv.set_logging_level('ERROR')
        
        reference_mesh = pv.read(args.reference)
        adaptive_mesh = pv.read(args.test_solution)

        # --- MODIFICA CHIAVE: Usiamo un KD-Tree per l'interpolazione del vicino più prossimo ---
        
        # 1. Estraiamo i punti e i dati da entrambe le mesh
        ref_points = reference_mesh.points
        # Assicuriamoci che l'array 'u' esista nella mesh di riferimento
        if 'u' not in reference_mesh.point_data:
            print(f"Errore: L'array di dati 'u' non è presente nella mesh di riferimento.", file=sys.stderr)
            sys.exit(1)
        ref_u_values = reference_mesh['u']
        
        adap_points = adaptive_mesh.points
        if 'u' not in adaptive_mesh.point_data:
            print(f"Errore: L'array di dati 'u' non è presente nella mesh di test.", file=sys.stderr)
            sys.exit(1)
        
        # 2. Costruiamo un KD-Tree dai punti della mesh di riferimento per una ricerca veloce
        print("Info: Costruzione del KD-Tree dai punti di riferimento...", file=sys.stderr)
        kdtree = KDTree(ref_points)
        print("Info: Costruzione completata.", file=sys.stderr)

        # 3. Per ogni punto nella mesh di test, troviamo l'indice del punto più vicino nella mesh di riferimento
        print("Info: Ricerca dei vicini più prossimi...", file=sys.stderr)
        _, indices = kdtree.query(adap_points, k=1)
        print("Info: Ricerca completata.", file=sys.stderr)

        # 4. Creiamo il nuovo array di dati usando i valori della mesh di riferimento agli indici trovati
        u_reference_on_adaptive_points = ref_u_values[indices]

        # 5. Aggiungiamo questo array alla nostra mesh di test con un nuovo nome
        adaptive_mesh['u_reference'] = u_reference_on_adaptive_points

        # Ora 'adaptive_mesh' contiene sia 'u' che 'u_reference' con la lunghezza corretta
        u_adaptive = adaptive_mesh['u']
        u_reference_sampled = adaptive_mesh['u_reference']

        # Calcola l'errore L2 integrato
        error_vector = u_adaptive - u_reference_sampled
        adaptive_mesh['error_squared'] = error_vector**2
        
        integration_result_table = adaptive_mesh.integrate_data('error_squared')
        integral_of_error_squared = integration_result_table['error_squared'][0]
        
        true_l2_error = np.sqrt(integral_of_error_squared)

        # Scrive il risultato sul file specificato
        with open(args.output_file, 'w') as f:
            f.write(str(true_l2_error))
        
        print(f"Info: Risultato scritto con successo in {args.output_file}", file=sys.stderr)

    except Exception as e:
        print(f"Errore fatale durante il calcolo: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
