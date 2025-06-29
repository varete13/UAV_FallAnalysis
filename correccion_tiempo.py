from pathlib import Path

def procesar_csv(ruta: Path):
    primeros_valores = {}
    resultado = []

    with ruta.open('r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue

            partes = linea.split(',')

            clave = partes[0]
            valor = float(partes[1])

            if clave not in primeros_valores:
                primeros_valores[clave] = valor
                nuevo_valor = 0.0
            else:
                nuevo_valor = valor - primeros_valores[clave]

            partes[1] = f"{nuevo_valor:.1f}"
            resultado.append(",".join(partes))

    ruta.write_text("\n".join(resultado), encoding='utf-8')

def procesar_todos_los_csv(directorio: Path):
    for archivo in directorio.glob("*.log"):
        procesar_csv(archivo)

procesar_todos_los_csv(Path('provolone/Alt_4_V_4'))