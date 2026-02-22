from datetime import date, timedelta

hoy = date(2026, 2, 22)
manana = hoy + timedelta(days=1)
lunes = manana - timedelta(days=manana.weekday())
sabado = lunes + timedelta(days=5)
lunes_sig = lunes + timedelta(weeks=1)
sabado_sig = lunes_sig + timedelta(days=5)

dias = ["Lun","Mar","Mie","Jue","Vie","Sab","Dom"]
print(f"Hoy:           {hoy} ({dias[hoy.weekday()]})")
print(f"Esta semana:   {manana} -> {sabado}")
print(f"Prox semana:   {lunes_sig} -> {sabado_sig}")
