import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Script from 'next/script'
import './globals.css'
import Providers from '@/components/shared/Providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: "Bird's-Eye Vision Platform",
  description: 'Computer vision platform for small-object analysis in aerial video',
}
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} dark`} suppressHydrationWarning>
        <Script
          id="strip-extension-hydration-attrs"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              try {
                document.querySelectorAll('[bis_skin_checked]').forEach(function (node) {
                  node.removeAttribute('bis_skin_checked');
                });
              } catch (error) {}
            `,
          }}
        />
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
