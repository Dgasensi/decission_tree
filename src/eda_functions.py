import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funcion de informacion agrupada de los dataframes
def df_info(df):
    df_info = pd.DataFrame({
        'nunique': df.nunique(),
        'nulls': df.isnull().sum(),
        'percent_nulls' : df.isnull().mean()*100,
        'Dtype': df.dtypes,
        'non_null': df.count(),
        'total_values': len(df)  
    })
    types_counter = df_info['Dtype'].value_counts()
    duplicated = df.duplicated().sum()
    display(df_info, types_counter)
    print(f'El dataframe tiene {df.shape[0]} filas y {df.shape[1]} columnas')
    print(f'Hay {duplicated} valores duplicados')

    ###############################################################################################

# Funcion para multiples boxplots
def multi_boxplot(dataframe, columns_list, n_rows, n_cols, fig_size=(12, 20)):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    axs = axs.flatten()  # Esto convierte la matriz de ejes en una lista plana, facilitando el acceso// sin esto buscar las posiciones en ejes era una locura
    
    for i, col in enumerate(columns_list):
        sns.boxplot(y=dataframe[col], ax=axs[i])
        axs[i].set_title(col)
        
    # Oculta los subgráficos adicionales que no se usan
    for ax in axs[len(columns_list):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    ###############################################################################################

# Funcion para multiples countplots

def multi_countplot(dataframe, columns_list, n_rows, n_cols, fig_size=(12, 20), percentage_fontsize=8):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    axs = axs.flatten()  # Esto convierte la matriz de ejes en una lista plana, facilitando el acceso
    
    for i, col in enumerate(columns_list):
        sns.countplot(x=dataframe[col], ax=axs[i])
        axs[i].set_title(col)
        axs[i].tick_params(axis='x', rotation=90)
    # Oculta los subgráficos adicionales que no se usan
    
        total = len(dataframe[col])

# Itera sobre cada barra del countplot
        for p in axs[i].patches:
            # Calcula el porcentaje y el texto a mostrar
            porcentaje = '{:.1f}%'.format(100 * p.get_height() / total)
            # Obtiene la posición en x y la altura de la barra(el valor)
            x = p.get_x() + p.get_width() / 2 
            y = p.get_height()
            # Añade el texto sobre la barra
            axs[i].annotate(porcentaje, (x, y), ha='center', va='bottom', fontsize=percentage_fontsize)

    for ax in axs[len(columns_list):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    ###############################################################################################

# Funcion para multiples histplot
def multi_histplot(dataframe, columns_list, n_rows, n_cols, fig_size=(12, 20)):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    axs = axs.flatten()
    
    for i, col in enumerate(columns_list):
        sns.histplot(x=dataframe[col], ax=axs[i], bins=50, kde=True)
        axs[i].set_title(col)
        
    # Oculta los subgráficos adicionales que no se usan
    for ax in axs[len(columns_list):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    ###############################################################################################

   
# Funcion para multiples boxplots comparativo
def multi_compare_box(dataframe, columns_list, n_rows, n_cols, fig_size=(12, 20)):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    axs = axs.flatten()  # Esto convierte la matriz de ejes en una lista plana, facilitando el acceso
    
    for i, col in enumerate(columns_list):
        sns.boxplot(x=dataframe['Success'], y=dataframe[col], ax=axs[i])
        axs[i].set_title(col)
        
    # Oculta los subgráficos adicionales que no se usan
    for ax in axs[len(columns_list):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    ###############################################################################################

def tasa_conversion(dataframe, var_predictora, var_target, type='line', order=None):
    x, y = var_predictora, var_target

    grupo = dataframe.groupby(x)[y].mean().mul(100).rename('tasa_conversion').reset_index()
    if type=='line':
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=var_predictora, y='tasa_conversion', data=grupo)
        plt.grid()
    elif type=='bar':
        plt.figure(figsize=(14, 6))
        sns.barplot(x=var_predictora, y='tasa_conversion', data=grupo, order=order)
        plt.grid()
    elif type=='scatter':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=var_predictora, y='tasa_conversion', data=grupo)
        plt.grid()

        ###############################################################################################

# Me ha costado la vida, pero he creado esta funcion para hacer target encoding en el futuro facilmente
def target_encoding(dataframe, column, target_column, smooth_coef=0):
    column, target_column = str(column), str(target_column)
    conteo_columna = dataframe.groupby(column)[target_column].count().reset_index(name= 'conteo')
    # Calculamos el promedio de LOGPRICE por vecindario
    promedio_columna = dataframe.groupby(column)[target_column].mean().reset_index(name= 'column_mean')
    # Unimos conteo y promedio en el mismo DF en base al vecindario
    final_df = promedio_columna.merge(conteo_columna, on=column)
    # Hallamos el promedio global 
    global_mean = dataframe[target_column].mean()
    # Asignamos coeficiente de suavizado
    m = smooth_coef
    # Calculamos precio suavizado para cada vecindario
    final_df[f'encoded_{column}'] = (final_df['conteo'] * final_df['column_mean'] + m * global_mean) / final_df['conteo'] + m
    # Creamos un diccionario para poder relacionar cada promedio con su valor original mas adelante
    encoded_value_dict = pd.Series(final_df[f'encoded_{column}'].values, index= final_df[column]).to_dict()
    # Fusionamos el dataframe original con el dataframe vecindarios en base a la columna 'neighbourhood'
    result_df = dataframe.merge(final_df[[column,str(f'encoded_{column}')]], on=column, how='left')
    return result_df

# ejemplo: df = target_encoding(df, 'neighbourhood', 'LOG_PRICE' )

    ###############################################################################################

def multi_countplot_compare(dataframe, columns_list, target, n_rows, n_cols, fig_size=(12, 20), percentage_fontsize=8):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    axs = axs.flatten()  # Esto convierte la matriz de ejes en una lista plana, facilitando el acceso

    for i, col in enumerate(columns_list):
        # Dibujamos el countplot
        sns.countplot(x=col, hue=target, data=dataframe, ax=axs[i])
        axs[i].set_title(col)
        axs[i].tick_params(axis='x', rotation=75)

        # Calculamos el total de respuestas para cada categoría en 'col'
        category_totals = (df.groupby('contact')['y'].value_counts()/df['contact'].value_counts())*100

        # Iteramos sobre cada conjunto de barras en el countplot
        for p in axs[i].patches:
            # Obtenemos la altura de la barra (número de respuestas)
            height = p.get_height()
            # Obtenemos la categoría a la que pertenece la barra
            category = p.get_x() + p.get_width() / 2.
            category = axs[i].get_xticks()[int(category)]
            category = axs[i].get_xticklabels()[category].get_text()

            # Calculamos el porcentaje basado en el total de respuestas para la categoría
            percentage = 100 * height / category_totals[category]

            # Añadimos la anotación del porcentaje sobre la barra
            axs[i].annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=percentage_fontsize)

    # Ocultamos los subgráficos adicionales que no se usan
    for ax in axs[len(columns_list):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
