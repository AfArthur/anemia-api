import requests
import json
import time

def testar_api():
    url = 'http://127.0.0.1:5000'
    
    print("\n🔍 Testando conexão com a API...")
    
    try:
        # 1. Testar endpoint de features
        resp_features = requests.get(f"{url}/features", timeout=5)
        resp_features.raise_for_status()  # Gera erro se status não for 200
        print("✅ Features do modelo:", resp_features.json())
        
        # 2. Preparar dados para predição
        dados = {
            "Number": 4,
            "Sex": 0,
            "Hb": 15.3,
            "%Blue pixel": 25.9,
            "%Green pixel": 30.8,
            "%Red Pixel": 43.2
        }
        
        print("\n📤 Enviando dados para predição:", dados)
        
        # 3. Fazer requisição de predição
        resp_predict = requests.post(
            f"{url}/predict",
            json=dados,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        # Verifica se a resposta contém JSON válido
        try:
            resultado = resp_predict.json()
            print("📊 Resultado da predição:", resultado)
        except json.JSONDecodeError:
            print("❌ Erro: A API não retornou um JSON válido")
            print("Resposta bruta:", resp_predict.text)
        
        # 4. Verificar predições no banco
        print("\n📦 Consultando todas as predições:")
        resp_consultar = requests.get(f"{url}/consultar", timeout=5)
        print(resp_consultar.json())
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro na requisição: {str(e)}")
    except Exception as e:
        print(f"⚠️ Erro inesperado: {str(e)}")

if __name__ == '__main__':
    testar_api()