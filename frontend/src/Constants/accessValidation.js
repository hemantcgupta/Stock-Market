import cookieStorage from './cookie-storage'

const accessValidation = (getParent, path) => {
    const menuData = JSON.parse(cookieStorage.getCookie('menuData'))
    let found = false;
    if (menuData.length > 0) {
        let parent = menuData.filter((data, i) => data.Name === getParent)
        let _ = parent.length > 0 ? parent[0].submenu.map((data) => data.Path.trim().includes(path) ? found = true : false) : null;
    }
    return found
}

const isRejectAlert = () => {
    const events = JSON.parse(cookieStorage.getCookie('events'))
    let found = false;
    if (events.length > 0) {
        events.filter((data, i) => data.event_code.toLowerCase().trim().includes('reject') ? found = true : false)
    }
    return found
}

const dashboardPath = () => {
    const menuData = JSON.parse(cookieStorage.getCookie('menuData'))
    let dashboard = menuData.filter((data, i) => data.Name === "Dashboard")
    let path = dashboard[0].submenu[0].Path;
    return path;
}

export default {
    accessValidation,
    isRejectAlert,
    dashboardPath,
};