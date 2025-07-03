# JBMD Website Deployment Guide for one.com

## Pre-Deployment Checklist

### ✅ Files Ready for Upload
- [ ] `index.html` (21.4KB) - Main website
- [ ] `styles.css` (21.8KB) - All styling
- [ ] `script.js` (9.1KB) - Interactive features
- [ ] `.htaccess` - Server configuration
- [ ] `robots.txt` - SEO crawler instructions
- [ ] `sitemap.xml` - SEO sitemap
- [ ] `site.webmanifest` - PWA manifest
- [ ] `README.md` - Documentation

### ⚠️ Optional Files (Create Later if Needed)
- `favicon.ico` (16x16 or 32x32 icon file)
- `favicon-16x16.png`
- `favicon-32x32.png` 
- `apple-touch-icon.png` (180x180)

---

## Step-by-Step Deployment Process

### Method 1: one.com Control Panel (Recommended)

1. **Login to one.com**
   - Go to https://www.one.com/
   - Click "Login" and enter your credentials

2. **Access File Manager**
   - In your control panel, look for "File Manager" or "Website Files"
   - Navigate to your domain's folder (usually named after your domain)
   - Enter the `public_html` or `www` directory

3. **Backup Current Site** (Important!)
   - Before uploading, download/backup your current files
   - Create a backup folder: `backup_old_site_YYYY-MM-DD`
   - Move existing files there

4. **Upload New Files**
   - Upload all 8 files to the root directory (`public_html` or `www`)
   - Ensure `.htaccess` is uploaded (may be hidden - enable "show hidden files")
   - Verify all files uploaded successfully

### Method 2: FTP Client (Alternative)

1. **Get FTP Details**
   - From one.com control panel, find FTP credentials
   - Server: Usually `ftp.yourdomain.co.uk` or similar
   - Username: Your hosting username
   - Password: Your hosting password

2. **Use FTP Client**
   - FileZilla (free): https://filezilla-project.org/
   - Connect using credentials above
   - Navigate to `public_html` or `www` folder
   - Upload all files

---

## Post-Deployment Verification

### Immediate Checks (Within 5 minutes)
1. **Visit Your Website**
   - Go to https://www.jbmd.co.uk
   - Check if new design loads properly
   - Test on mobile device

2. **Test Key Functions**
   - [ ] Navigation links work (About, Experience, Contact)
   - [ ] Email links open correctly
   - [ ] Contact forms/buttons function
   - [ ] Mobile responsiveness works

3. **Check Load Speed**
   - Should load in under 3 seconds
   - No broken images or missing styles

### SEO Verification (Within 24 hours)
1. **Google Search Console**
   - Submit new sitemap: `https://www.jbmd.co.uk/sitemap.xml`
   - Check for crawl errors

2. **Social Media Preview**
   - Test link sharing on LinkedIn
   - Verify Open Graph tags display correctly

---

## Troubleshooting Common Issues

### ❌ "Internal Server Error" (500 Error)
- **Cause**: Usually `.htaccess` file issues
- **Solution**: Remove `.htaccess` temporarily, test site, then re-upload

### ❌ Styles Not Loading
- **Cause**: File permissions or caching
- **Solution**: 
  - Check file permissions (should be 644)
  - Clear browser cache (Ctrl+F5)
  - Wait 10-15 minutes for server cache

### ❌ Site Shows "Under Construction"
- **Cause**: Old index file taking precedence
- **Solution**: Ensure `index.html` is in root directory, delete old index files

### ❌ Favicon Not Showing
- **Cause**: Missing icon files (this is normal initially)
- **Solution**: Create simple favicon.ico file or ignore for now

---

## DNS & SSL Configuration

### SSL Certificate (HTTPS)
- one.com usually provides free SSL
- Enable in control panel under "SSL" or "Security"
- May take 24-48 hours to activate

### Domain Settings
- Ensure www.jbmd.co.uk points to your hosting
- Verify both www and non-www versions work

---

## Emergency Rollback Plan

If something goes wrong:

1. **Restore from Backup**
   - Use the backup files you created
   - Upload them back to public_html

2. **Contact one.com Support**
   - 24/7 support available
   - Reference your hosting account details

---

## File Permissions Guide

Set these permissions after upload:
- `index.html`: 644
- `styles.css`: 644  
- `script.js`: 644
- `.htaccess`: 644
- `robots.txt`: 644
- `sitemap.xml`: 644
- `site.webmanifest`: 644

---

## Success Indicators

✅ **Website Successfully Deployed When:**
- New professional design loads at www.jbmd.co.uk
- All sections display properly (Hero, About, Experience, Skills, Contact)
- Contact email links work: jill.adams@jbmd.co.uk
- Mobile version displays correctly
- Page loads in under 3 seconds
- No console errors in browser developer tools

---

## Next Steps After Deployment

1. **Monitor for 48 hours** - Check for any issues
2. **Update Google My Business** with new website design
3. **Test email links** from different devices
4. **Share with colleagues** for feedback
5. **Update business cards/materials** if needed

---

**Need Help?** The website is designed to be robust, but if you encounter issues during deployment, the most common solutions are:
1. Clear browser cache
2. Wait 15 minutes for server updates
3. Check file permissions
4. Verify all files uploaded to correct directory

Good luck with the deployment! The new website will significantly enhance your professional presence and prospect engagement.