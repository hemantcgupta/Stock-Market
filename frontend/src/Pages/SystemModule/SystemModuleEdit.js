import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import cookieStorage from '../../Constants/cookie-storage';
import Loader from '../../Component/Loader';
import api from '../../API/api'
import Access from "../../Constants/accessValidation";
import './systemModule.scss';
import _ from 'lodash';
import Swal from 'sweetalert2';


const apiServices = new APIServices();


class SystemModuleEdit extends Component {
    constructor(props) {
        super(props);
        this.state = {
            moduleName: '',
            modulePath: null,
            visibility: '',
            moduleType: '',
            eventCode: null,
            eventCodeErrmsg: '',
            moduleNameErrmsg: '',
            modulePathErrmsg: '',
            disable: true

        };

    }

    componentDidMount() {
        if (Access.accessValidation('System Modules', 'systemModuleEdit')) {
            if (this.props.location.state) {
                let data = this.props.location.state.data
                this.setState({
                    moduleName: data.Module_Name,
                    modulePath: data.Module_Path,
                    visibility: data.visibility,
                    moduleType: data.module_type,
                    eventCode: data.event_code
                })
            }
        } else {
            this.props.history.push('/404')
        }
    }

    handleChange(e) {
        this.setState({ [e.target.id]: e.target.value }, this._validate(e))
    }

    _validate = (e) => {
        let id = e.target.id;
        let value = e.target.value.trim();
        let msg = ''

        if (id === 'moduleName') {
            msg = value === '' ? 'Please enter module name' : ''
            this.setState({ moduleNameErrmsg: msg })
        } else if (id === 'modulePath') {
            msg = value === '' ? 'Please enter module path' : ''
            this.setState({ modulePathErrmsg: msg })
        } else if (id === 'eventCode') {
            msg = value === '' ? 'Enter event Code' : ''
            this.setState({ eventCodeErrmsg: msg })
        }

        setTimeout(() => {
            this.formValid()
        }, 600);
    }

    formValid = () => {
        const { moduleNameErrmsg, modulePathErrmsg, moduleName, modulePath, moduleType, eventCodeErrmsg, eventCode, visibility } = this.state;
        let errorCheck = _.isEmpty(moduleNameErrmsg && modulePathErrmsg)
        let emptyDataCheck = !_.isEmpty(moduleName && modulePath)
        if (moduleType === 'event') {
            errorCheck = _.isEmpty(moduleNameErrmsg && modulePathErrmsg && eventCodeErrmsg)
            emptyDataCheck = !_.isEmpty(moduleName && eventCode)
        }
        if (errorCheck && emptyDataCheck && visibility) {
            this.setState({ disable: false })
        } else {
            this.setState({ disable: true })
        }
    }

    selectVisibility = (event) => {
        this.setState({
            visibility: event.target.value,
        }, () => this.formValid());
    }

    updateSystemModule = () => {
        const data = this.props.location.state.data
        var postData = {
            id: data.id,
            name: this.state.moduleName,
            path: this.state.modulePath === '' ? null : this.state.modulePath,
            visibility: this.state.visibility,
            parentid: data.parentid,
            event_code: this.state.eventCode,
            module_type: data.module_type,
            create_date: data.create_date,
            is_deleted: data.is_deleted
        };

        api.put(`rest/updatemodules/${data.id}/`, postData)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `System Module updated successfully`,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.hideAlertSuccess();
                    this.getUpdatedMenus();
                })

            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to update system module`,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                })
            })
    }

    getUpdatedMenus = () => {
        api.get(`config`)
            .then((response) => {
                if (response) {
                    if (response.data.length > 0) {
                        cookieStorage.createCookie('menuData', JSON.stringify(response.data.menus), 1);
                        cookieStorage.createCookie('events', JSON.stringify(response.data.events), 1);
                    }
                }
            })
            .catch((err) => {
            })
    }

    hideAlertSuccess() {
        this.props.history.push(`/systemModuleSearch`)
    }

    render() {

        return (
            <div>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className='system-module'>
                    <div className='add'>
                        <h2>Edit System Module</h2>
                    </div>
                    <div className='module-form'>
                        <div class="form-group">
                            <label for="moduleName">Module Name</label>
                            <input type="text" class="form-control" id="moduleName" value={this.state.moduleName} onChange={(e) => this.handleChange(e)} />
                            <p>{this.state.moduleNameErrmsg}</p>
                        </div>

                        <div class="form-group" >
                            <label for="modulePath">
                                Visibility
                            </label>
                            <select onChange={(e) => this.selectVisibility(e)}>
                                <option selected={this.state.visibility === 'show'} value="show">Show</option>
                                <option selected={this.state.visibility === 'hide'} value="hide">Hide</option>
                            </select>
                        </div>

                        {this.state.moduleType === 'page' ?
                            <div class="form-group">
                                <label for="modulePath">Module Path</label>
                                <input type="text" class="form-control" id="modulePath" value={this.state.modulePath} onChange={(e) => this.handleChange(e)} />
                                <p>{this.state.modulePathErrmsg}</p>
                            </div> :
                            <div class="form-group"  >
                                <label for="eventCode">Event Code</label>
                                <input type="text" class="form-control" id="eventCode" value={this.state.eventCode} onChange={(e) => this.handleChange(e)} />
                                <p>{this.state.eventCodeErrmsg}</p>
                            </div>}

                        <div className='action-btn'>
                            <button type="button" className="btn btn-danger" disabled={this.state.disable} onClick={() => this.updateSystemModule()}>Update</button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default SystemModuleEdit;