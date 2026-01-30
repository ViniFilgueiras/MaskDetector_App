import os

# Esse código converte o arquivo .tflite para um arquivo .h para ser usado na ESP32-CAM

# Configurações de arquivo
TFLITE_FILE = "detector_leve_int8.tflite"
HEADER_FILE = "model_data.h"

def convert_tflite_to_h(tflite_path, header_path):
    # Verificando se o arquivo tflite existe
    if not os.path.exists(tflite_path):
        print(f"Erro: O arquivo {tflite_path} não foi encontrado!")
        return

    # Lendo o modelo binário
    with open(tflite_path, 'rb') as f:
        tflite_content = f.read()

    # Criando o conteúdo do arquivo .h
    hex_lines = []
    for i, byte in enumerate(tflite_content):
        # Formatando cada byte como 0x00
        line_byte = f"0x{byte:02x}"
        hex_lines.append(line_byte)

    # Organizando em linhas de 12 bytes para facilitar a leitura
    formatted_hex = ""
    for i in range(0, len(hex_lines), 12):
        formatted_hex += "  " + ", ".join(hex_lines[i:i+12]) + ",\n"

    # Criando o template final
    header_content = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Tamanho do modelo: {len(tflite_content) / 1024:.2f} KB
unsigned int g_model_len = {len(tflite_content)};
alignas(8) const unsigned char g_model[] = {{
{formatted_hex.rstrip(',\\n')}
}};

#endif // MODEL_DATA_H
"""

    # Salvando o arquivo
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"Sucesso! Arquivo '{header_path}' gerado.")
    print(f"Tamanho: {len(tflite_content)} bytes.")

# Executa a conversão
convert_tflite_to_h(TFLITE_FILE, HEADER_FILE)
