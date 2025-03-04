"""
Google Colab と ローカル環境の両方で動作するための互換性モジュール
"""

import os
import sys

def is_running_in_colab():
    """
    現在の実行環境がGoogle Colabかどうかを判定します
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_environment():
    """
    実行環境に応じた設定を行います
    """
    if is_running_in_colab():
        # Colabでの設定
        print("Google Colab環境で実行しています")
        
        # Google Driveのマウント
        from google.colab import drive
        drive.mount('/content/drive/')
        
        # 必要なパッケージのインストール
        import subprocess
        subprocess.run(["pip", "install", "google-cloud-secret-manager", "openai", "anthropic", "pydub", "httpx==0.26.0", "openai==1.54.0"])
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"])
        
        # デフォルトのパス設定
        base_input_dir = "/content/drive/MyDrive/開発/Whisper/Input"
        base_output_dir = "/content/drive/MyDrive/開発/Whisper/Output"
    else:
        # ローカル環境での設定
        print("ローカル環境で実行しています")
        
        # 必要なパッケージが既にインストールされていることを確認
        required_packages = ["openai", "anthropic", "pydub", "httpx", "moviepy"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"以下のパッケージをインストールする必要があります: {', '.join(missing_packages)}")
            print("pip install " + " ".join(missing_packages))
            print("インストール後に再度実行してください")
            sys.exit(1)
        
        # ローカル環境でのデフォルトパス設定
        # カレントディレクトリにInput/Outputフォルダを作成
        current_dir = os.getcwd()
        base_input_dir = os.path.join(current_dir, "Input")
        base_output_dir = os.path.join(current_dir, "Output")
        
        # 必要なディレクトリの作成
        os.makedirs(base_input_dir, exist_ok=True)
        os.makedirs(base_output_dir, exist_ok=True)
    
    return {
        "base_input_dir": base_input_dir,
        "base_output_dir": base_output_dir
    }

def get_api_keys():
    """
    APIキーを取得します。
    Colab環境ではGoogle Cloud Secret Managerから、
    ローカル環境では環境変数または.envファイルから取得します。
    """
    if is_running_in_colab():
        # Colabでの認証とAPIキー取得
        from google.colab import auth
        from google.cloud import secretmanager
        
        # Google Colabでの認証
        auth.authenticate_user()
        
        # Google Cloud Secret Managerから環境変数にAPIキーを設定
        # プロジェクトIDをユーザーに入力してもらう
        project_id = input("Google Cloud プロジェクトIDを入力してください: ")
        if not project_id:
            print("プロジェクトIDが入力されていません。APIキーの取得に失敗します。")
            return
        
        def access_secret(project_id, secret_name, version='latest'):
            """Google Cloud Secret Managerから秘密情報を取得する"""
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
            response = client.access_secret_version(request={"name": name})
            payload = response.payload.data.decode("UTF-8")
            return payload
        
        # OpenAIのAPIキー
        openai_secret_name = "OPENAI_API_KEY"
        openai_api_key = access_secret(project_id, openai_secret_name)
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            print("OpenAI API key has been set as an environment variable.")
        else:
            print("Failed to set the OpenAI API key as an environment variable.")
        
        # 後処理用のClaudeのAPIキー
        anthropic_secret_name = "ANTHROPIC_API_KEY"
        anthropic_api_key = access_secret(project_id, anthropic_secret_name)
        
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            print("Anthropic API key has been set as an environment variable.")
        else:
            print("Failed to set the Anthropic API key as an environment variable.")
    else:
        # ローカル環境でのAPIキー取得
        # .envファイルからの読み込みを試みる
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print(".envファイルを使用するにはpython-dotenvをインストールしてください: pip install python-dotenv")
        
        # 環境変数の確認
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not openai_api_key:
            print("警告: OPENAI_API_KEYが設定されていません。")
            print("環境変数として設定するか、.envファイルに追加してください。")
        else:
            print("OpenAI API keyが環境変数から読み込まれました。")
        
        if not anthropic_api_key:
            print("警告: ANTHROPIC_API_KEYが設定されていません。")
            print("環境変数として設定するか、.envファイルに追加してください。")
        else:
            print("Anthropic API keyが環境変数から読み込まれました。")
