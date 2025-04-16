from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuração inicial
app = Flask(__name__)

# --- Função para reset seguro do banco ---
def reset_database():
    db_path = 'predicoes.db'
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print("♻️ Banco de dados antigo removido")
        except PermissionError:
            print("⚠️ Usando banco existente (não pôde ser removido)")

# --- Carregamento do Modelo ---
try:
    modelo = joblib.load('modelo_random_forest.joblib')
    print("✅ Modelo carregado com sucesso!")
    print("🔍 Features esperadas:", list(modelo.feature_names_in_))
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {str(e)}")
    modelo = None

# --- Configuração do Banco de Dados ---
reset_database()
engine = create_engine('sqlite:///predicoes.db')
Base = declarative_base()

class Predicao(Base):
    __tablename__ = 'predicoes'
    id = Column(Integer, primary_key=True)
    Number = Column(Integer)
    Sex = Column(Integer)
    Hb = Column(Float)
    blue_pixel = Column(Float)  # %Blue pixel
    green_pixel = Column(Float) # %Green pixel
    red_pixel = Column(Float)   # %Red Pixel
    resultado = Column(Integer)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- Endpoints ---
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Endpoint de verificação de saúde da API"""
    return jsonify({
        "status": "online",
        "modelo_carregado": bool(modelo),
        "features": list(modelo.feature_names_in_) if modelo else None,
        "database": "operacional"
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not modelo:
        return jsonify({"erro": "Modelo não carregado"}), 500
    
    sessao = None
    try:
        dados = request.get_json()
        
        # Verificação das colunas
        required = ['Number', 'Sex', 'Hb', '%Blue pixel', '%Green pixel', '%Red Pixel']
        if not all(col in dados for col in required):
            return jsonify({
                "erro": "Colunas faltando",
                "colunas_necessarias": required,
                "colunas_recebidas": list(dados.keys())
            }), 400

        # Cria DataFrame garantindo a ordem correta
        df = pd.DataFrame([{
            'Number': dados['Number'],
            'Sex': dados['Sex'],
            'Hb': dados['Hb'],
            '%Blue pixel': dados['%Blue pixel'],
            '%Green pixel': dados['%Green pixel'],
            '%Red Pixel': dados['%Red Pixel']
        }])[modelo.feature_names_in_]  # Ordem das features do modelo

        # Predição
        resultado = int(modelo.predict(df)[0])

        # Persistência no banco
        sessao = Session()
        nova_pred = Predicao(
            Number=dados['Number'],
            Sex=dados['Sex'],
            Hb=dados['Hb'],
            blue_pixel=dados['%Blue pixel'],
            green_pixel=dados['%Green pixel'],
            red_pixel=dados['%Red Pixel'],
            resultado=resultado
        )
        sessao.add(nova_pred)
        sessao.commit()

        return jsonify({
            "predicao": resultado,
            "features_utilizadas": list(modelo.feature_names_in_)
        })

    except Exception as e:
        if sessao:
            sessao.rollback()
        return jsonify({
            "erro": str(e),
            "tipo_erro": type(e).__name__
        }), 500
    finally:
        if sessao:
            sessao.close()

@app.route('/features', methods=['GET'])
def features():
    if not modelo:
        return jsonify({"erro": "Modelo não carregado"}), 500
    return jsonify({
        "features": list(modelo.feature_names_in_),
        "detalhes": {
            "Number": "Número de identificação",
            "Sex": "Gênero (0=Feminino, 1=Masculino)",
            "Hb": "Nível de hemoglobina",
            "%Blue pixel": "Percentual de pixels azuis",
            "%Green pixel": "Percentual de pixels verdes",
            "%Red Pixel": "Percentual de pixels vermelhos"
        }
    })

@app.route('/consultar', methods=['GET'])
def consultar():
    sessao = Session()
    try:
        predicoes = sessao.query(Predicao).all()
        return jsonify([{
            "id": p.id,
            "Number": p.Number,
            "Sex": p.Sex,
            "Hb": p.Hb,
            "%Blue pixel": p.blue_pixel,
            "%Green pixel": p.green_pixel,
            "%Red Pixel": p.red_pixel,
            "resultado": p.resultado,
            "data_criacao": str(p.id)  # Timestamp implícito
        } for p in predicoes])
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        sessao.close()

# --- Configurações Finais ---
@app.teardown_appcontext
def shutdown_session(exception=None):
    if 'engine' in globals():
        engine.dispose()
        print("🔌 Conexões com o banco encerradas")

@app.route('/')
def home():
    return jsonify({
        "message": "Bem-vindo à API de Predição de Anemia",
        "endpoints": {
            "healthcheck": "/healthcheck (GET)",
            "prediction": "/predict (POST)",
            "features": "/features (GET)",
            "consult": "/consultar (GET)"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)