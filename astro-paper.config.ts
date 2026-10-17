import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://sureshr-dev.github.io/",
    title: "Suresh Rajendiran",
    description:
      "Site Reliability Engineer with a habit of learning out loud — notes and write-ups from the journey.",
    author: "Suresh Rajendiran",
    profile: "https://sureshr-dev.github.io/",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Kolkata",
    dir: "ltr",
  },
  posts: {
    perPage: 4,
    perIndex: 4,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: false,
    },
    search: "pagefind",
  },
  socials: [
    { name: "github",   url: "https://github.com/sureshr-dev" },
    { name: "linkedin", url: "https://www.linkedin.com/in/sureshrajendran/" },
  ],
  shareLinks: [
    { name: "whatsapp", url: "https://wa.me/?text=" },
    { name: "facebook", url: "https://www.facebook.com/sharer.php?u=" },
    { name: "x",        url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "pinterest", url: "https://pinterest.com/pin/create/button/?url=" },
    { name: "mail",     url: "mailto:?subject=See%20this%20post&body=" },
  ],
});