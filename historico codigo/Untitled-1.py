# %%
import Utils
import pandas as pd



# %%
Speeds=Utils.log_attitude("logs_test1/2025-03-27 18-19-21.txt")

# %%


# %%
SP=pd.DataFrame(Speeds)
# SP['Fecha'] = pd.to_datetime(SP['Fecha']).dt.strftime('%d/%m/%Y')


# %%
SP.plot(x='Hora',y='roll',title='Velocidad del avión',xlabel='Hora')

# %%



