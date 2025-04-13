from __future__ import annotations

import argparse
import os
import random
import textwrap
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta

from fpdf import FPDF

PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

DOCTORS = [
    "Dr. Ana Morales",
    "Dr. Juan Pérez",
    "Dr. Luisa García",
    "Dr. Pablo López",
    "Dr. Camila Díaz",
    "Dr. Andrés Torres",
    "Dr. Laura Ruiz",
    "Dr. Sergio Romero",
    "Dr. Beatriz Castro",
    "Dr. Tomás Fernández",
    "Dr. Elena Vargas",
    "Dr. Ricardo Soto",
    "Dr. Patricia Méndez",
    "Dr. Francisco Navarro",
    "Dr. Claudia Ríos",
    "Dr. Diego Silva",
    "Dr. Valeria Paredes",
    "Dr. Gabriel Ortega",
    "Dr. Natalia Herrera",
    "Dr. Martín Cruz",
]

PATIENTS = [
    "Carlos Sánchez",
    "María González",
    "Pedro Ramírez",
    "Lucía Herrera",
    "Sofía Castro",
    "Alejandro Martínez",
    "Isabella Romero",
    "Mateo Delgado",
    "Valentina Ríos",
    "Daniel Ortega",
    "Emilia Navarro",
    "Julián Vargas",
    "Gabriela Peña",
    "Santiago León",
    "Martina Silva",
    "David Paredes",
    "Renata Morales",
    "Nicolás Rojas",
    "Victoria Cruz",
    "Lucas Herrera",
    "Andrea Mendoza",
    "Felipe Torres",
    "Camila Díaz",
    "Sebastián Ruiz",
    "Valeria Soto",
    "Diego Castro",
    "Antonella Pérez",
    "Maximiliano López",
    "Florencia Morales",
    "Bruno Silva",
    "Catalina Ríos",
    "Emilio Vargas",
    "Agustina Torres",
    "Thiago González",
    "Mía Herrera",
    "Benjamín Castro",
    "Emma Sánchez",
    "Dylan Martínez",
    "Olivia Ramírez",
    "Noah Delgado",
    "Sophia Ortega",
    "Liam Navarro",
    "Isabella Méndez",
    "Ethan Cruz",
    "Mía Soto",
    "Aiden Silva",
    "Luna Vargas",
    "Lucas Torres",
    "Emma Herrera",
    "Oliver Castro",
    "Sofía Morales",
    "William Ríos",
]

MEDICINES = [
    "Paracetamol 500mg",
    "Ibuprofeno 600mg",
    "Amoxicilina 875mg",
    "Omeprazol 20mg",
    "Loratadina 10mg",
    "Metformina 850mg",
    "Enalapril 10mg",
    "Salbutamol Inhalador",
    "Atorvastatina 20mg",
    "Azitromicina 500mg",
    "Losartán 50mg",
    "Simvastatina 20mg",
    "Clopidogrel 75mg",
    "Amlodipino 5mg",
    "Levotiroxina 50mcg",
    "Metoprolol 50mg",
    "Pantoprazol 40mg",
    "Ciprofloxacino 500mg",
    "Diclofenaco 50mg",
    "Prednisona 5mg",
    "Sertralina 50mg",
    "Fluoxetina 20mg",
    "Alprazolam 0.5mg",
    "Clonazepam 2mg",
    "Diazepam 5mg",
    "Hidroclorotiazida 25mg",
    "Furosemida 40mg",
    "Espironolactona 25mg",
    "Digoxina 0.25mg",
    "Warfarina 5mg",
    "Insulina Glargina",
    "Insulina Aspart",
    "Metformina XR 1000mg",
    "Glimepirida 2mg",
    "Pioglitazona 15mg",
    "Sitagliptina 100mg",
    "Empagliflozina 10mg",
    "Canagliflozina 100mg",
    "Dapagliflozina 10mg",
    "Liraglutida 1.2mg",
    "Semaglutida 0.5mg",
    "Dulaglutida 0.75mg",
]

PROBLEMS = [
    "dolor de cabeza persistente",
    "tos y congestión nasal",
    "dolor abdominal leve",
    "fiebre alta",
    "presión arterial elevada",
    "dolor en el pecho",
    "alergia estacional",
    "mareos frecuentes",
    "dolor de espalda",
    "insomnio",
    "dolor de garganta",
    "dolor muscular generalizado",
    "dolor en las articulaciones",
    "náuseas y vómitos",
    "diarrea",
    "estreñimiento",
    "acidez estomacal",
    "reflujo gastroesofágico",
    "dolor de oído",
    "visión borrosa",
    "dolor en los ojos",
    "zumbido en los oídos",
    "dolor en las piernas",
    "hinchazón en los tobillos",
    "dolor en los pies",
    "dolor en las manos",
    "hormigueo en las extremidades",
    "debilidad muscular",
    "fatiga crónica",
    "ansiedad",
    "depresión",
    "estrés",
    "palpitaciones",
    "dificultad para respirar",
    "tos con flema",
    "sangrado nasal",
    "dolor al orinar",
    "incontinencia urinaria",
    "dolor lumbar",
    "dolor cervical",
    "rigidez en el cuello",
    "dolor de hombro",
    "dolor en la rodilla",
    "dolor en la cadera",
    "dolor en el codo",
    "dolor en la muñeca",
    "dolor en los dedos",
    "dolor en el talón",
]

DIAGNOSES = [
    "Resfriado común",
    "Gripe (Influenza)",
    "Faringitis estreptocócica",
    "Bronquitis aguda",
    "Neumonía",
    "Asma",
    "Enfermedad Pulmonar Obstructiva Crónica (EPOC)",
    "Rinitis alérgica",
    "Sinusitis",
    "Otitis media",
    "Infección del tracto urinario (ITU)",
    "Gastroenteritis viral",
    "Gastritis",
    "Enfermedad por reflujo gastroesofágico (ERGE)",
    "Síndrome del intestino irritable (SII)",
    "Apendicitis",
    "Colecistitis",
    "Hipertensión arterial",
    "Diabetes mellitus tipo 2",
    "Hipotiroidismo",
    "Hipertiroidismo",
    "Anemia por deficiencia de hierro",
    "Migraña",
    "Cefalea tensional",
    "Vértigo posicional paroxístico benigno (VPPB)",
    "Lumbalgia",
    "Ciática",
    "Artrosis (Osteoartritis)",
    "Artritis reumatoide",
    "Gota",
    "Tendinitis",
    "Bursitis",
    "Fibromialgia",
    "Depresión mayor",
    "Trastorno de ansiedad generalizada",
    "Trastorno de pánico",
    "Insomnio crónico",
    "Dermatitis atópica (Eczema)",
    "Psoriasis",
    "Acné vulgar",
    "Infección por hongos (Micosis cutánea)",
    "Celulitis infecciosa",
    "Varicela",
    "Herpes zóster",
    "Conjuntivitis",
    "Cataratas",
    "Glaucoma",
    "Degeneración macular asociada a la edad (DMAE)",
]

RECOMMENDATIONS = [
    "Se recomienda reposo y abundante hidratación.",
    "Iniciar tratamiento con el medicamento prescripto durante 7 días.",
    "Realizar análisis de sangre y volver con los resultados.",
    "Controlar la presión diariamente y volver en una semana.",
    "Evitar alimentos irritantes y descansar.",
    "Programar una ecografía para descartar complicaciones.",
    "Iniciar tratamiento sintomático y seguir observación.",
    "Referir al especialista en cardiología para evaluación.",
    "Suspender actividad física por 5 días.",
    "Seguir dieta blanda y observar evolución.",
    "Iniciar tratamiento con antibióticos por 10 días.",
    "Realizar radiografía de tórax para descartar neumonía.",
    "Controlar la glucosa en ayunas y posprandial.",
    "Iniciar terapia física para fortalecimiento muscular.",
    "Programar consulta con oftalmólogo para evaluación.",
    "Realizar audiometría para evaluar la audición.",
    "Iniciar tratamiento con antiinflamatorios por 5 días.",
    "Controlar el peso y la presión arterial semanalmente.",
    "Realizar electrocardiograma para evaluación cardíaca.",
    "Iniciar tratamiento con antidepresivos y terapia psicológica.",
    "Programar resonancia magnética para evaluación neurológica.",
    "Realizar prueba de esfuerzo para evaluar capacidad cardíaca.",
    "Iniciar tratamiento con antihistamínicos por 7 días.",
    "Controlar la función tiroidea con análisis de sangre.",
    "Realizar ecografía abdominal para evaluación de órganos.",
    "Iniciar tratamiento con broncodilatadores y corticoides.",
    "Programar consulta con nutricionista para plan alimentario.",
    "Realizar densitometría ósea para evaluar osteoporosis.",
    "Iniciar tratamiento con relajantes musculares por 3 días.",
    "Controlar la función renal con análisis de orina.",
    "Realizar tomografía computada para evaluación detallada.",
    "Iniciar tratamiento con antivirales por 5 días.",
    "Programar consulta con dermatólogo para evaluación cutánea.",
    "Realizar prueba de alergias para identificar alérgenos.",
    "Iniciar tratamiento con ansiolíticos y terapia de relajación.",
    "Controlar la función hepática con análisis de sangre.",
    "Realizar endoscopia para evaluación gastrointestinal.",
    "Iniciar tratamiento con antifúngicos por 14 días.",
    "Programar consulta con traumatólogo para evaluación ósea.",
]


@dataclass
class Consultation:
    patient: str
    doctor: str
    medicine: str
    problem: str
    diagnosis: str
    recommendation: str
    date: datetime

    @staticmethod
    def safe_text(text: str, width: int = 100) -> str:
        return "\n".join(textwrap.wrap(text, width))

    @staticmethod
    def generate() -> Consultation:
        return Consultation(
            patient=random.choice(PATIENTS),
            doctor=random.choice(DOCTORS),
            medicine=random.choice(MEDICINES) if random.random() < 0.5 else "",
            problem=random.choice(PROBLEMS),
            diagnosis=random.choice(DIAGNOSES) if DIAGNOSES else "Diagnóstico pendiente",
            recommendation=random.choice(RECOMMENDATIONS),
            date=datetime.now() - timedelta(days=random.randint(0, 365)),
        )

    def to_pdf(self, path: str = PDF_DIR) -> str:
        pdf = FPDF()
        pdf.set_margins(20, 20, 20)  # Left, Top, Right margins
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Calculate effective width for multi_cell
        effective_width = pdf.w - pdf.l_margin - pdf.r_margin

        pdf.cell(effective_width, 10, txt="Informe de Consulta Médica", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(effective_width, 10, txt=f"Fecha de consulta: {self.date.strftime('%d/%m/%Y')}", ln=True)
        pdf.cell(effective_width, 10, txt=f"Paciente: {self.patient}", ln=True)
        pdf.cell(effective_width, 10, txt=f"Médico: {self.doctor}", ln=True)
        pdf.ln(10)
        pdf.multi_cell(effective_width, 10, txt=self.safe_text(f"Motivo de la consulta: {self.problem}", width=60))
        pdf.multi_cell(effective_width, 10, txt=self.safe_text(f"Diagnóstico: {self.diagnosis}", width=60))
        pdf.multi_cell(effective_width, 10, txt=self.safe_text(f"Recomendación: {self.recommendation}", width=60))
        if self.medicine:
            pdf.multi_cell(effective_width, 10, txt=self.safe_text(f"Medicamento prescripto: {self.medicine}", width=60))
        filename = os.path.join(path, f"{self.patient.replace(' ', '_')}_{self.date.strftime('%d-%m-%Y')}.pdf")
        pdf.output(filename)
        print(f"PDF generado: {filename}")
        return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de PDFs de consultas médicas")
    parser.add_argument("-n", "--number", type=int, default=1, help="Número de PDFs a generar (por defecto: 1)")
    args = parser.parse_args()
    for i in range(0, args.number):
        print(f"Creating {i + 1} consultation out of {args.number}")
        Consultation.generate().to_pdf()
