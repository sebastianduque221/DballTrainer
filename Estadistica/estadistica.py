import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import tempfile
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

# -----------------------------
# CLASES BASE Y COMPONENTES
# -----------------------------

class DataLoader:
    """Responsible for loading and validating data"""
    @staticmethod
    def load_json_data(file_path: Path) -> pd.DataFrame:
        """Load and clean JSON data into DataFrame"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                raise ValueError("El archivo JSON está vacío.")
            df = pd.DataFrame(data)
            df.columns = df.columns.str.strip().str.replace(" ", ".", regex=False)
            return df.dropna(how='all')
        except Exception as e:
            raise RuntimeError(f"Error al cargar datos: {e}")

class ReportGenerator(ABC):
    """Abstract base class for report generation"""
    @abstractmethod
    def generate(self, data: pd.DataFrame, output_dir: Path, analysis_type: str) -> None:
        pass

class PDFReportGenerator(ReportGenerator):
    """Concrete implementation for PDF report generation"""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def generate(self, data: pd.DataFrame, output_dir: Path, analysis_type: str) -> None:
        """Generate comprehensive PDF report"""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_title(f"Analisis de {analysis_type.capitalize()}")
        pdf.set_author("Sistema de Analisis Deportivo")
        
        self._add_cover_page(pdf, analysis_type)
        self._add_statistical_summary(pdf, data)
        self._add_correlation_matrix(pdf, data)
        self._add_histograms(pdf, data, analysis_type)
        
        pdf_path = output_dir / f"Reporte_{analysis_type.capitalize()}.pdf"
        pdf.output(pdf_path)
        print(f"\n✅ Reporte PDF generado en: {pdf_path}")

    def _add_cover_page(self, pdf: FPDF, analysis_type: str) -> None:
        """Add cover page to PDF"""
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 24)
        pdf.cell(0, 40, "Reporte de Analisis Deportivo", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(20)
        
        pdf.set_font("Helvetica", 'B', 18)
        pdf.cell(0, 20, analysis_type.capitalize(), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        
        pdf.set_font("Helvetica", '', 14)
        pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(30)
        
        pdf.set_font("Helvetica", 'I', 12)
        pdf.multi_cell(0, 10, 
                      "Este reporte contiene un analisis detallado de los parametros tecnicos evaluados durante la sesion de entrenamiento.")

    def _add_statistical_summary(self, pdf: FPDF, data: pd.DataFrame) -> None:
        """Add statistical summary to PDF"""
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, "Resumen Estadistico", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        col_width = 40
        row_height = 10
        
        for col in numeric_cols[:10]:
            stats = data[col].describe()
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(col_width, row_height, col, 1)
            pdf.set_font("Helvetica", '', 12)
            pdf.cell(col_width, row_height, f"Media: {stats['mean']:.2f}", 1)
            pdf.cell(col_width, row_height, f"Desv: {stats['std']:.2f}", 1)
            pdf.cell(col_width, row_height, f"Min/Max: {stats['min']:.1f}/{stats['max']:.1f}", 1)
            pdf.ln(row_height)

    def _add_correlation_matrix(self, pdf: FPDF, data: pd.DataFrame) -> None:
        """Add correlation matrix to PDF"""
        if data.empty or len(data.select_dtypes(include=['number']).columns) < 2:
            return
            
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            data.select_dtypes(include=['number']).corr(), 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f",
            vmin=-1, 
            vmax=1,
            center=0
        )
        plt.title("Matriz de Correlación", pad=20)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_path = os.path.join(self.temp_dir, "matriz_correlacion.png")
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, "Matriz de Correlacion", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(img_path, x=10, y=30, w=180)
        pdf.ln(120)

    def _add_histograms(self, pdf: FPDF, data: pd.DataFrame, analysis_type: str) -> None:
        """Add histograms to PDF"""
        histogram_imgs = self._generate_histograms(data, analysis_type)
        if not histogram_imgs:
            return
            
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, "Distribuciones Clave", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        for img_path in histogram_imgs:
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.image(img_path, x=10, w=180)
            pdf.ln(85)

    def _generate_histograms(self, data: pd.DataFrame, analysis_type: str) -> List[str]:
        """Generate histogram images"""
        if data.empty:
            return []
            
        relevant_columns = {
            "ataque": ["Angulo.Codo.Izq", "Angulo.Codo.Der", "Velocidad.Angular.Codo.Izq"],
            "bloqueo": ["Angulo.Brazo.Izq", "Angulo.Brazo.Der", "Altura.Bloqueo.Izq"],
            "recibo": ["Angulo.Tronco", "Profundidad.Sentadilla", "Distancia.Entre.Pies"],
            "saque": ["Angulo.Codo", "Altura.Brazo", "Alineacion.Hombro"],
            "colocador": ["Angulo.Codo.Izq", "Angulo.Codo.Der", "Angulo.Rodilla.Izq"]
        }.get(analysis_type, data.select_dtypes(include=['number']).columns[:6])
        
        img_paths = []
        for col in relevant_columns:
            if col not in data.columns or data[col].dropna().empty:
                continue
                
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], kde=True, color='dodgerblue', bins=15)
            plt.title(f"Distribución de {col}", pad=15)
            plt.xlabel(col)
            plt.grid(alpha=0.3)
            
            img_path = os.path.join(self.temp_dir, f"hist_{col}.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()
            img_paths.append(img_path)
            
        return img_paths

class AnalysisStrategy(ABC):
    """Strategy interface for different analysis types"""
    @abstractmethod
    def generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        pass

class AttackAnalysisStrategy(AnalysisStrategy):
    def generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        recommendations = [
            "RECOMENDACIONES PARA ATAQUE:",
            "----------------------------------------"
        ]
        
        if all(col in data.columns for col in ["Angulo.Codo.Izq", "Angulo.Codo.Der"]):
            angle_diff = abs(data["Angulo.Codo.Izq"] - data["Angulo.Codo.Der"]).mean()
            evaluation = "OPTIMA" if angle_diff <= 15 else "A MEJORAR"
            recommendations.append(
                f"- Diferencia media entre codos: {angle_diff:.1f} grados (Recomendado <15) - {evaluation}"
            )
            
        if all(col in data.columns for col in ["Velocidad.Angular.Codo.Izq", "Velocidad.Angular.Codo.Der"]):
            speed_diff = abs(data["Velocidad.Angular.Codo.Izq"] - data["Velocidad.Angular.Codo.Der"]).mean()
            evaluation = "BALANCEADA" if speed_diff <= 1.0 else "DESIGUAL"
            recommendations.append(
                f"- Diferencia de velocidad angular: {speed_diff:.2f} rad/s (Recomendado <1.0) - {evaluation}"
            )
            
        if "Ataque.Valido" in data.columns:
            valid_percentage = data["Ataque.Valido"].mean() * 100
            evaluation = "EXCELENTE" if valid_percentage >= 85 else "ACEPTABLE" if valid_percentage >= 70 else "A MEJORAR"
            recommendations.append(
                f"- Porcentaje de ataques validos: {valid_percentage:.1f}% (Objetivo 85% o mas) - {evaluation}"
            )
            
        return recommendations

class BlockAnalysisStrategy(AnalysisStrategy):
    def generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        recommendations = [
            "RECOMENDACIONES PARA BLOQUEO:",
            "----------------------------------------"
        ]
        if all(col in data.columns for col in ["Angulo.Brazo.Izq", "Angulo.Brazo.Der"]):
            angle_diff = abs(data["Angulo.Brazo.Izq"] - data["Angulo.Brazo.Der"]).mean()
            evaluation = "SIMÉTRICO" if angle_diff <= 10 else "MEJORAR SIMETRÍA"
            recommendations.append(
                f"- Diferencia media entre brazos: {angle_diff:.1f} grados (Recomendado <10) - {evaluation}"
            )
        if "Bloqueo.Válido" in data.columns:
            valid_percentage = data["Bloqueo.Válido"].mean() * 100
            evaluation = "EXCELENTE" if valid_percentage >= 85 else "ACEPTABLE" if valid_percentage >= 70 else "A MEJORAR"
            recommendations.append(
                f"- Porcentaje de bloqueos válidos: {valid_percentage:.1f}% (Objetivo 85% o más) - {evaluation}"
            )
        if "Simetría" in data.columns:
            sim_percentage = data["Simetría"].mean() * 100
            recommendations.append(
                f"- Simetría en bloqueos: {sim_percentage:.1f}% de los bloqueos fueron simétricos"
            )
        return recommendations

class ReceiveAnalysisStrategy(AnalysisStrategy):
    def generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        recommendations = [
            "RECOMENDACIONES PARA RECIBO:",
            "----------------------------------------"
        ]
        if "Angulo.Tronco" in data.columns:
            avg_trunk = data["Angulo.Tronco"].mean()
            evaluation = "ÓPTIMO" if 80 <= avg_trunk <= 100 else "AJUSTAR POSTURA"
            recommendations.append(
                f"- Ángulo promedio del tronco: {avg_trunk:.1f}° (Ideal 80-100°) - {evaluation}"
            )
        if "Profundidad.Sentadilla" in data.columns:
            avg_squat = data["Profundidad.Sentadilla"].mean()
            recommendations.append(
                f"- Profundidad promedio de sentadilla: {avg_squat:.2f}"
            )
        if "Distancia.Entre.Pies" in data.columns:
            avg_dist = data["Distancia.Entre.Pies"].mean()
            recommendations.append(
                f"- Distancia promedio entre pies: {avg_dist:.2f}"
            )
        return recommendations

class ServeAnalysisStrategy(AnalysisStrategy):
    def generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        recommendations = [
            "RECOMENDACIONES PARA SAQUE:",
            "----------------------------------------"
        ]
        if "Angulo.Codo" in data.columns:
            avg_angle = data["Angulo.Codo"].mean()
            evaluation = "ÓPTIMO" if avg_angle > 90 else "MEJORAR EXTENSIÓN"
            recommendations.append(
                f"- Ángulo promedio del codo: {avg_angle:.1f}° (Ideal >90°) - {evaluation}"
            )
        if "Altura.Brazo" in data.columns:
            avg_height = data["Altura.Brazo"].mean()
            recommendations.append(
                f"- Altura promedio del brazo: {avg_height:.2f}"
            )
        if "Saque.Válido" in data.columns:
            valid_percentage = data["Saque.Válido"].mean() * 100
            evaluation = "EXCELENTE" if valid_percentage >= 85 else "ACEPTABLE" if valid_percentage >= 70 else "A MEJORAR"
            recommendations.append(
                f"- Porcentaje de saques válidos: {valid_percentage:.1f}% (Objetivo 85% o más) - {evaluation}"
            )
        return recommendations

class SetterAnalysisStrategy(AnalysisStrategy):
    def generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        recommendations = [
            "RECOMENDACIONES PARA COLOCADOR:",
            "----------------------------------------"
        ]
        if all(col in data.columns for col in ["Angulo.Codo.Izq", "Angulo.Codo.Der"]):
            avg_left = data["Angulo.Codo.Izq"].mean()
            avg_right = data["Angulo.Codo.Der"].mean()
            recommendations.append(
                f"- Ángulo promedio codo izquierdo: {avg_left:.1f}°"
            )
            recommendations.append(
                f"- Ángulo promedio codo derecho: {avg_right:.1f}°"
            )
        if "Angulo.Rodilla.Izq" in data.columns:
            avg_knee = data["Angulo.Rodilla.Izq"].mean()
            recommendations.append(
                f"- Ángulo promedio rodilla izquierda: {avg_knee:.1f}°"
            )
        return recommendations

class AnalysisContext:
    """Context class that uses the strategy pattern"""
    def __init__(self, strategy: AnalysisStrategy):
        self._strategy = strategy
        
    def generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        return self._strategy.generate_recommendations(data)

# -----------------------------
# MAIN ANALYSIS CLASS
# -----------------------------

class AnalisisEstadistico:
    """Main class for performing sports technique analysis"""
    def __init__(self, json_path: Path, analysis_type: str):
        self.json_path = json_path
        self.analysis_type = analysis_type.lower()
        self.data = None
        self.output_dir = Path(f"Estadistica/Resultados/{self.analysis_type.capitalize()}")
        self.report_generator = PDFReportGenerator()
        
    def load_data(self) -> None:
        """Load and validate the input data"""
        self.data = DataLoader.load_json_data(self.json_path)
        print("Datos cargados correctamente.")
        print("Columnas:", list(self.data.columns))

    def create_output_directory(self) -> None:
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Carpeta de resultados: {self.output_dir}")

    def perform_analysis(self) -> None:
        """Execute the complete analysis workflow"""
        print("\n" + "="*50)
        print(f"INICIANDO ANALISIS DE {self.analysis_type.upper()}")
        print("="*50)
        
        self.load_data()
        self.create_output_directory()
        
        # Create appropriate strategy based on analysis type
        strategy = self._create_analysis_strategy()
        analysis_context = AnalysisContext(strategy)
        recommendations = analysis_context.generate_recommendations(self.data)
        
        # Generate PDF report
        self.report_generator.generate(self.data, self.output_dir, self.analysis_type)
        
        print("\n" + "="*50)
        print("ANALISIS COMPLETADO CON EXITO")
        print("="*50)
        print("\n".join(recommendations))

    def _create_analysis_strategy(self) -> AnalysisStrategy:
        """Factory method to create appropriate strategy"""
        strategies = {
            "ataque": AttackAnalysisStrategy(),
            "bloqueo": BlockAnalysisStrategy(),
            "recibo": ReceiveAnalysisStrategy(),
            "saque": ServeAnalysisStrategy(),
            "colocador": SetterAnalysisStrategy()
        }
        
        if self.analysis_type not in strategies:
            raise ValueError(f"Tipo de analisis no soportado: {self.analysis_type}")
            
        return strategies[self.analysis_type]

# -----------------------------
# UI AND MAIN PROGRAM
# -----------------------------

def get_json_file_path() -> Optional[Path]:
    """Prompt user for JSON file path via consola"""
    path = input("Ingrese la ruta del archivo JSON de datos: ").strip()
    if not path or not os.path.isfile(path):
        print("Archivo no encontrado.")
        return None
    return Path(path)

def get_analysis_type() -> Optional[str]:
    """Get analysis type from user input"""
    print("\nTIPOS DE ANALISIS DISPONIBLES:")
    print("1. Ataque\n2. Bloqueo\n3. Recibo\n4. Saque\n5. Colocador")
    try:
        option = int(input("\nSeleccione el tipo de analisis (1-5): "))
        types = ["ataque", "bloqueo", "recibo", "saque", "colocador"]
        if 1 <= option <= 5:
            return types[option-1]
    except ValueError:
        pass
    return None

def main():
    print("SISTEMA DE ANALISIS DE TECNICA DEPORTIVA")
    print("="*50 + "\n")
    print("Bienvenido al sistema de analisis de tecnica deportiva.")
    
    json_path = get_json_file_path()
    if not json_path:
        print("No se seleccionó ningún archivo. Saliendo...")
        return
        
    analysis_type = get_analysis_type()
    if not analysis_type:
        print("Opción no válida. Saliendo...")
        return
        
    try:
        analyzer = AnalisisEstadistico(json_path, analysis_type)
        analyzer.perform_analysis()
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    main()