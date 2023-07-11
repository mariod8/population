import { createApp } from "vue"
import App from "./App.vue"
import { OhVueIcon, addIcons } from "oh-vue-icons"
import { FaGithub } from "oh-vue-icons/icons"

import "./assets/main.css"

addIcons(FaGithub)

const app = createApp(App)

app.component("v-icon", OhVueIcon)
app.mount("#app")
