# JBMD Website Design System

*Based on FSCompliance brand alignment and professional financial services requirements*

---

## Brand Values & Design Principles

### **Core Brand Values**
- **Technical Excellence** - Deep expertise, precision, reliability
- **Professional Independence** - Objective, unbiased, trustworthy
- **Forward Innovation** - AI specialization, future-focused thinking
- **Regulatory Intelligence** - Compliance expertise, industry knowledge
- **Client-Focused** - Solution-oriented, practical insights

### **Design Principles**
- **Authoritative yet Approachable** - Professional credibility with human connection
- **Clean & Modern** - Contemporary design reflecting technology expertise
- **Information Clarity** - Easy scanning for busy financial professionals
- **Conservative Innovation** - Progressive but trustworthy for financial industry
- **Prospect-Optimized** - Maximum impact for email referrals

---

## Color Palette

### **Primary Colors**
```css
/* Deep Professional Blue - Primary brand color */
--primary-blue: #1e3a8a;      /* rgb(30, 58, 138) */
--primary-blue-light: #3b82f6; /* rgb(59, 130, 246) */
--primary-blue-dark: #1e40af;  /* rgb(30, 64, 175) */

/* Sophisticated Charcoal - Secondary */
--charcoal: #374151;           /* rgb(55, 65, 81) */
--charcoal-light: #6b7280;     /* rgb(107, 114, 128) */
--charcoal-dark: #1f2937;      /* rgb(31, 41, 55) */
```

### **Accent Colors**
```css
/* Success/Growth Green */
--accent-green: #059669;       /* rgb(5, 150, 105) */
--accent-green-light: #10b981; /* rgb(16, 185, 129) */

/* Warning/Attention Gold */
--accent-gold: #d97706;        /* rgb(217, 119, 6) */
--accent-gold-light: #f59e0b;  /* rgb(245, 158, 11) */
```

### **Neutral Palette**
```css
/* Background & Layout */
--white: #ffffff;              /* Pure white */
--gray-50: #f9fafb;           /* Light background */
--gray-100: #f3f4f6;          /* Section backgrounds */
--gray-200: #e5e7eb;          /* Borders */
--gray-300: #d1d5db;          /* Subtle borders */
--gray-800: #1f2937;          /* Dark text */
--gray-900: #111827;          /* Headings */
```

---

## Typography System

### **Font Selection**
```css
/* Primary Font - Professional Sans-Serif */
--font-primary: 'Inter', 'Segoe UI', 'Roboto', sans-serif;

/* Headings Font - Slightly more distinctive */
--font-headings: 'Inter', 'Segoe UI', 'Roboto', sans-serif;

/* Monospace - For technical content if needed */
--font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
```

### **Typography Scale**
```css
/* Heading Sizes */
--text-5xl: 3rem;      /* 48px - Hero headline */
--text-4xl: 2.25rem;   /* 36px - Page title */
--text-3xl: 1.875rem;  /* 30px - Section headers */
--text-2xl: 1.5rem;    /* 24px - Subsection headers */
--text-xl: 1.25rem;    /* 20px - Large text */

/* Body Sizes */
--text-lg: 1.125rem;   /* 18px - Large body text */
--text-base: 1rem;     /* 16px - Regular body text */
--text-sm: 0.875rem;   /* 14px - Small text */
--text-xs: 0.75rem;    /* 12px - Fine print */
```

### **Font Weights**
```css
--font-thin: 100;
--font-light: 300;
--font-normal: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
--font-extrabold: 800;
```

---

## Layout System

### **Container & Grid**
```css
/* Container Widths */
--container-sm: 640px;   /* Small screens */
--container-md: 768px;   /* Medium screens */
--container-lg: 1024px;  /* Large screens */
--container-xl: 1280px;  /* Extra large screens */
--container-2xl: 1536px; /* 2X large screens */

/* Grid System */
--grid-cols-12: repeat(12, minmax(0, 1fr));
--grid-cols-6: repeat(6, minmax(0, 1fr));
--grid-cols-4: repeat(4, minmax(0, 1fr));
--grid-cols-3: repeat(3, minmax(0, 1fr));
--grid-cols-2: repeat(2, minmax(0, 1fr));
```

### **Spacing Scale**
```css
/* Spacing System (based on 4px grid) */
--space-1: 0.25rem;    /* 4px */
--space-2: 0.5rem;     /* 8px */
--space-3: 0.75rem;    /* 12px */
--space-4: 1rem;       /* 16px */
--space-5: 1.25rem;    /* 20px */
--space-6: 1.5rem;     /* 24px */
--space-8: 2rem;       /* 32px */
--space-10: 2.5rem;    /* 40px */
--space-12: 3rem;      /* 48px */
--space-16: 4rem;      /* 64px */
--space-20: 5rem;      /* 80px */
--space-24: 6rem;      /* 96px */
```

---

## Component Styles

### **Buttons**
```css
/* Primary Button */
.btn-primary {
  background-color: var(--primary-blue);
  color: var(--white);
  padding: var(--space-3) var(--space-6);
  border-radius: 0.375rem;
  font-weight: var(--font-medium);
  font-size: var(--text-base);
  transition: all 0.2s ease;
  border: none;
  cursor: pointer;
}

.btn-primary:hover {
  background-color: var(--primary-blue-dark);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
}

/* Secondary Button */
.btn-secondary {
  background-color: transparent;
  color: var(--primary-blue);
  border: 2px solid var(--primary-blue);
  padding: var(--space-3) var(--space-6);
  border-radius: 0.375rem;
  font-weight: var(--font-medium);
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background-color: var(--primary-blue);
  color: var(--white);
}
```

### **Cards**
```css
.card {
  background-color: var(--white);
  border-radius: 0.75rem;
  padding: var(--space-6);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid var(--gray-200);
  transition: all 0.2s ease;
}

.card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.card-header {
  font-size: var(--text-xl);
  font-weight: var(--font-semibold);
  color: var(--gray-900);
  margin-bottom: var(--space-3);
}
```

### **Navigation**
```css
.nav-link {
  color: var(--charcoal);
  font-weight: var(--font-medium);
  padding: var(--space-2) var(--space-4);
  text-decoration: none;
  transition: color 0.2s ease;
}

.nav-link:hover {
  color: var(--primary-blue);
}

.nav-link.active {
  color: var(--primary-blue);
  font-weight: var(--font-semibold);
}
```

---

## Visual Elements

### **Shadows**
```css
/* Shadow System */
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07);
--shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
--shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.15);
--shadow-2xl: 0 25px 50px rgba(0, 0, 0, 0.25);
```

### **Border Radius**
```css
--radius-sm: 0.25rem;   /* 4px */
--radius-md: 0.375rem;  /* 6px */
--radius-lg: 0.5rem;    /* 8px */
--radius-xl: 0.75rem;   /* 12px */
--radius-2xl: 1rem;     /* 16px */
--radius-full: 50%;     /* Circular */
```

### **Transitions**
```css
--transition-fast: 0.15s ease;
--transition-normal: 0.2s ease;
--transition-slow: 0.3s ease;
```

---

## Responsive Breakpoints

```css
/* Mobile First Approach */
@media (min-width: 640px) { /* sm */ }
@media (min-width: 768px) { /* md */ }
@media (min-width: 1024px) { /* lg */ }
@media (min-width: 1280px) { /* xl */ }
@media (min-width: 1536px) { /* 2xl */ }
```

---

## Brand Alignment Notes

### **FSCompliance Consistency**
- **Primary blue** aligns with FSCompliance's professional, trustworthy identity
- **Clean typography** reflects technical precision and attention to detail
- **Generous spacing** conveys quality and sophistication
- **Subtle animations** show modern approach without being flashy
- **Professional shadows** add depth while maintaining conservative aesthetic

### **Financial Industry Standards**
- **Conservative color palette** appropriate for financial services
- **High contrast ratios** for accessibility and readability
- **Clear hierarchy** for easy information scanning
- **Professional imagery approach** (if images are used)
- **Mobile-first responsive** for modern professional workflow

---

*This design system provides the foundation for a sharp, professional website that builds immediate credibility with prospects while maintaining perfect brand alignment with FSCompliance values.*