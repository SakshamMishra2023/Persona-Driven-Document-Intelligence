import json
import os
import re
from datetime import datetime
from pathlib import Path
import pypdf
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

class EnhancedDocumentIntelligence:
    def __init__(self):
        # Load the lightweight sentence transformer model
        print("Loading sentence transformer model...")
        try:
            # Set offline mode to prevent any network calls
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with page information"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                pages_text = []
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        pages_text.append({
                            'page': page_num,
                            'text': text.strip()
                        })
                
                return pages_text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return []
    
    def extract_sections(self, text, page_num):
        """Improved section extraction with better logic"""
        sections = []
        
        # Clean the text first
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        lines = text.split('\n')
        
        current_section = []
        current_title = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Enhanced heading detection
            is_heading = False
            
            # Pattern 1: Major headings (longer, descriptive titles)
            if (len(line.split()) >= 3 and len(line.split()) <= 15 and 
                (line.istitle() or line.isupper()) and 
                not line.startswith('•') and not line.startswith('-') and
                len(line) > 20 and len(line) < 200):
                is_heading = True
            
            # Pattern 2: Numbered sections
            elif re.match(r'^\d+\.?\s+[A-Z]', line) and len(line.split()) >= 2:
                is_heading = True
            
            # Pattern 3: All caps headers (but not too short)
            elif (line.isupper() and len(line.split()) >= 2 and len(line.split()) <= 10 and 
                  len(line) > 10 and len(line) < 100):
                is_heading = True
            
            # Pattern 4: Lines ending with colon (topic introducers)
            elif (line.endswith(':') and len(line.split()) >= 2 and len(line.split()) <= 8 and
                  len(line) > 15 and not line.startswith('•')):
                is_heading = True
            
            # Skip bullet points and short fragments as headings
            if (line.startswith('•') or line.startswith('-') or line.startswith('*') or
                len(line) < 15 or len(line.split()) <= 2):
                is_heading = False
            
            if is_heading:
                # Save previous section if it has substantial content
                if current_title and current_section and len(' '.join(current_section)) > 100:
                    sections.append({
                        'title': current_title,
                        'content': ' '.join(current_section),
                        'page': page_num
                    })
                
                # Start new section
                current_title = line.rstrip(':')
                current_section = []
            else:
                current_section.append(line)
        
        # Add last section if substantial
        if current_title and current_section and len(' '.join(current_section)) > 100:
            sections.append({
                'title': current_title,
                'content': ' '.join(current_section),
                'page': page_num
            })
        
        # If no good sections found, create meaningful chunks
        if not sections:
            # Split text into larger, meaningful chunks
            words = text.split()
            chunk_size = 300  # Larger chunks for better context
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text) > 200:  # Only substantial chunks
                    # Create a meaningful title from first sentence
                    first_sentence = chunk_text.split('.')[0]
                    if len(first_sentence) > 100:
                        first_sentence = first_sentence[:100] + "..."
                    
                    title = first_sentence if first_sentence else f"Page {page_num} Content Part {i//chunk_size + 1}"
                    
                    sections.append({
                        'title': title,
                        'content': chunk_text,
                        'page': page_num
                    })
        
        return sections
    
    def generate_role_context(self, persona_role):
        """Generate role-specific context using semantic similarity"""
        # Create a comprehensive role description query
        role_query = f"What does a {persona_role} do? What are their key responsibilities, skills, and focus areas?"
        
        # Define a comprehensive vocabulary of professional terms and concepts
        professional_vocabulary = [
            # Planning & Strategy
            "planning", "strategy", "objectives", "goals", "roadmap", "timeline", "scheduling",
            "coordination", "organization", "management", "execution", "implementation",
            
            # Analysis & Research
            "analysis", "research", "data", "insights", "trends", "metrics", "evaluation",
            "assessment", "investigation", "findings", "statistics", "performance",
            
            # Communication & Collaboration
            "communication", "collaboration", "teamwork", "presentation", "reporting",
            "consultation", "advice", "recommendations", "guidance", "support",
            
            # Problem Solving
            "problem solving", "solutions", "optimization", "improvement", "efficiency",
            "best practices", "innovation", "creativity", "troubleshooting",
            
            # Domain-specific terms
            "travel", "tourism", "itinerary", "destinations", "accommodation", "transportation",
            "budget", "logistics", "booking", "reservations", "activities", "attractions",
            "finance", "accounting", "investment", "risk", "compliance", "audit",
            "marketing", "sales", "customer", "client", "service", "satisfaction",
            "technology", "software", "systems", "development", "design", "architecture",
            "healthcare", "medical", "patient", "treatment", "diagnosis", "care",
            "education", "training", "learning", "curriculum", "instruction", "assessment",
            "legal", "law", "regulations", "contracts", "compliance", "litigation",
            "operations", "processes", "workflow", "quality", "standards", "procedures"
        ]
        
        try:
            # Get embeddings for role query and all vocabulary terms
            role_embedding = self.model.encode([role_query])
            vocab_embeddings = self.model.encode(professional_vocabulary)
            
            # Calculate similarity between role and each vocabulary term
            similarities = cosine_similarity(role_embedding, vocab_embeddings)[0]
            
            # Get top relevant terms (above threshold)
            threshold = 0.3  # Adjust as needed
            relevant_terms = []
            
            for i, similarity in enumerate(similarities):
                if similarity > threshold:
                    relevant_terms.append((professional_vocabulary[i], similarity))
            
            # Sort by relevance and take top terms
            relevant_terms.sort(key=lambda x: x[1], reverse=True)
            top_terms = [term[0] for term in relevant_terms[:15]]  # Top 15 terms
            
            # Create context string
            context = " ".join(top_terms)
            return context
            
        except Exception as e:
            print(f"Error generating role context: {e}")
            # Fallback: extract key terms from the role name itself
            role_words = persona_role.lower().split()
            return " ".join(role_words + ["professional", "work", "tasks", "responsibilities"])
    
    def create_enhanced_persona_query(self, persona_role, job_task):
        """Create a more comprehensive semantic query using dynamic context generation"""
        # Generate role-specific context dynamically
        context = self.generate_role_context(persona_role)
        
        # Create comprehensive query
        query = f"""A {persona_role} working on {job_task}. This involves {context}. 
        The person needs comprehensive information, practical details, specific recommendations, 
        step-by-step guidance, and actionable insights to successfully complete this task."""
        
        return query
    
    def calculate_semantic_relevance(self, text, persona_query):
        """Enhanced semantic relevance calculation"""
        if not text.strip() or len(text) < 50:
            return 0.0
        
        try:
            # Get embeddings
            text_embedding = self.model.encode([text])
            query_embedding = self.model.encode([persona_query])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, text_embedding)[0][0]
            
            # Boost score for longer, more detailed content
            length_bonus = min(len(text) / 2000, 0.2)  # Up to 0.2 bonus for long content
            
            # Boost score for content with practical information
            practical_keywords = ['tip', 'recommend', 'should', 'best', 'how to', 'guide', 'step', 'plan']
            practical_score = sum(1 for keyword in practical_keywords if keyword in text.lower()) * 0.02
            
            final_score = float(similarity) + length_bonus + practical_score
            return min(final_score, 1.0)
            
        except Exception as e:
            print(f"Error calculating semantic relevance: {e}")
            return 0.0
    
    def rank_sections(self, sections, persona_query):
        """Enhanced ranking with better scoring"""
        if not sections:
            return []
        
        print(f"Ranking {len(sections)} sections based on enhanced semantic relevance...")
        
        # Calculate relevance scores
        for i, section in enumerate(sections):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing section {i+1}/{len(sections)}")
            
            # Combine title and content with title weighted more heavily
            title_weighted = f"{section['title']} {section['title']} {section['title']}"  # Triple weight for title
            full_text = f"{title_weighted} {section['content']}"
            
            section['relevance_score'] = self.calculate_semantic_relevance(full_text, persona_query)
        
        # Sort by relevance score (highest first)
        ranked_sections = sorted(sections, key=lambda x: x['relevance_score'], reverse=True)
        
        print("Section ranking completed!")
        return ranked_sections
    
    def extract_meaningful_content(self, content, max_sentences=2):
        """Extract meaningful sentences, not fragments"""
        if len(content) <= 300:
            return [content]
        
        # Split into sentences properly
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30]
        
        if len(sentences) <= max_sentences:
            return sentences
        
        try:
            # Get embeddings for all sentences
            sentence_embeddings = self.model.encode(sentences)
            
            # Calculate average embedding (centroid)
            centroid = np.mean(sentence_embeddings, axis=0)
            
            # Calculate similarity of each sentence to the centroid
            similarities = cosine_similarity([centroid], sentence_embeddings)[0]
            
            # Get indices of top sentences
            top_indices = np.argsort(similarities)[-max_sentences:][::-1]
            
            # Return top sentences in original order, but ensure they're substantial
            selected_sentences = []
            for i in sorted(top_indices):
                sentence = sentences[i]
                if len(sentence) > 50:  # Only substantial sentences
                    selected_sentences.append(sentence)
            
            # If we don't have enough substantial sentences, add more
            if len(selected_sentences) < max_sentences:
                for sentence in sentences:
                    if sentence not in selected_sentences and len(sentence) > 30:
                        selected_sentences.append(sentence)
                        if len(selected_sentences) >= max_sentences:
                            break
            
            return selected_sentences[:max_sentences]
            
        except Exception as e:
            print(f"Error in content extraction: {e}")
            # Fallback: return first few substantial sentences
            substantial_sentences = [s for s in sentences if len(s) > 50]
            return substantial_sentences[:max_sentences] if substantial_sentences else sentences[:max_sentences]
    
    def process_documents(self, input_data):
        """Main processing function - processes ALL documents together"""
        documents = input_data['documents']
        persona = input_data['persona']
        job_to_be_done = input_data['job_to_be_done']
        
        # Create enhanced semantic query
        persona_query = self.create_enhanced_persona_query(persona['role'], job_to_be_done['task'])
        print(f"Enhanced persona query: {persona_query}")
        
        # Extract all sections from ALL documents
        all_sections = []
        
        print("Extracting sections from all documents...")
        for doc in documents:
            pdf_path = f"input/PDFs/{doc['filename']}"
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF not found: {pdf_path}")
                continue
                
            print(f"Processing: {doc['filename']}")
            pages_text = self.extract_text_from_pdf(pdf_path)
            
            for page_data in pages_text:
                sections = self.extract_sections(page_data['text'], page_data['page'])
                
                for section in sections:
                    section['document'] = doc['filename']
                    section['document_title'] = doc['title']
                    all_sections.append(section)
        
        print(f"Total sections extracted from all documents: {len(all_sections)}")
        
        # Rank ALL sections using enhanced semantic similarity
        ranked_sections = self.rank_sections(all_sections, persona_query)
        
        # Take top sections but ensure diversity across documents
        top_sections = []
        document_count = {}
        
        # First pass: ensure at least 2-3 sections per document
        for section in ranked_sections:
            doc_name = section['document']
            if document_count.get(doc_name, 0) < 3:  # Max 3 sections per document initially
                top_sections.append(section)
                document_count[doc_name] = document_count.get(doc_name, 0) + 1
            
            if len(top_sections) >= 15:
                break
        
        # Second pass: fill remaining slots with highest scoring sections
        if len(top_sections) < 15:
            for section in ranked_sections:
                if section not in top_sections:
                    top_sections.append(section)
                    if len(top_sections) >= 15:
                        break
        
        # Build output in the required format
        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in documents],
                "persona": persona['role'],
                "job_to_be_done": job_to_be_done['task'],
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        print("Generating final output...")
        
        # Process extracted sections
        for rank, section in enumerate(top_sections, 1):
            output["extracted_sections"].append({
                "document": section['document'],
                "section_title": section['title'],
                "importance_rank": rank,
                "page_number": section['page']
            })
            
            # Extract meaningful content pieces for subsection analysis
            content_pieces = self.extract_meaningful_content(section['content'], max_sentences=2)
            
            for piece in content_pieces:
                if piece.strip() and len(piece.strip()) > 50:  # Only substantial content
                    output["subsection_analysis"].append({
                        "document": section['document'],
                        "refined_text": piece.strip(),
                        "page_number": section['page']
                    })

        return output

def main():
    """Main function that processes ALL documents together and creates ONE output"""
    print("Initializing Enhanced Document Intelligence System...")
    
    try:
        doc_intel = EnhancedDocumentIntelligence()
        
        # Read input configuration
        input_json_path = 'input/input.json'
        
        if os.path.exists(input_json_path):
            print("Reading input data from input.json...")
            with open(input_json_path, 'r') as f:
                input_data = json.load(f)
        else:
            print("No input.json found, creating default configuration...")
            # Find all PDFs in input directory
            pdf_files = [f for f in os.listdir('input/PDFs') if f.endswith('.pdf')]
            
            if not pdf_files:
                print("No PDF files found in /app/input directory!")
                return
            
            # Create default configuration
            input_data = {
                "documents": [{"filename": pdf, "title": pdf.replace('.pdf', '')} for pdf in pdf_files],
                "persona": {"role": "Professional Analyst"},
                "job_to_be_done": {"task": "Extract and analyze key information from documents"}
            }
        
        # Process ALL documents together
        print("Processing all documents together...")
        result = doc_intel.process_documents(input_data)
        
        # Write single combined output
        print("Writing combined output...")
        with open('output/output.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("Processing completed successfully!")
        print(f"Analyzed {len(result['metadata']['input_documents'])} documents together")
        print(f"Extracted {len(result['extracted_sections'])} most relevant sections")
        print(f"Generated {len(result['subsection_analysis'])} refined text pieces")
        print("Combined analysis saved to output.json")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        # Create error output
        error_output = {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        with open('output/output.json', 'w') as f:
            json.dump(error_output, f, indent=2)
        raise

if __name__ == "__main__":
    main()