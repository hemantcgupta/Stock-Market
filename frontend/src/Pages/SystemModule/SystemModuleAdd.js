import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import api from '../../API/api'
import Access from "../../Constants/accessValidation";
import './systemModule.scss';
import _ from 'lodash';
import Swal from 'sweetalert2';


const apiServices = new APIServices();


class SystemModuleAdd extends Component {
    constructor(props) {
        super(props);
        this.state = {
            moduleName: '',
            modulePath: '',
            moduleNameErrmsg: '',
            modulePathErrmsg: '',
            disable: true,
            systemModules: [],
            selectParentID: null,
            isParentModule: null,
            visibility: 'show',
            isEvent: false,
            eventCode: null,
            eventCodeErrmsg: '',
        };

    }

    componentDidMount() {
        if (Access.accessValidation('System Modules', 'systemModuleAdd')) {
            this.getSystemModulelist();
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
            msg = value === '' ? 'Enter module name' : ''
            this.setState({ moduleNameErrmsg: msg })
        } else if (id === 'modulePath') {
            msg = value === '' ? 'Enter module path' : ''
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
        const { moduleNameErrmsg, modulePathErrmsg, moduleName, modulePath, eventCodeErrmsg, eventCode, isEvent, selectParentID } = this.state;
        let errorCheck = _.isEmpty(moduleNameErrmsg && modulePathErrmsg)
        let emptyDataCheck = !_.isEmpty(moduleName && modulePath)
        if (isEvent) {
            errorCheck = _.isEmpty(moduleNameErrmsg && modulePathErrmsg && eventCodeErrmsg)
            emptyDataCheck = !_.isEmpty(moduleName && eventCode)
        }
        if (errorCheck && emptyDataCheck) {
            this.setState({ disable: false })
        } else {
            this.setState({ disable: true })
        }
    }

    selectParentID = (event) => {
        let { name, value } = event.target;
        this.setState({
            selectParentID: value !== "" && value !== undefined ? value : null,
        });
    }

    selectVisibility = (event) => {
        this.setState({
            visibility: event.target.value,
        });
    }

    selectModuleType = (event) => {
        let { name, value } = event.target;
        this.setState({ isParentModule: null, isEvent: false, modulePath: '' })
        if (value !== "" && value !== undefined) {
            if (value === "parentmodule") {
                this.setState({
                    isParentModule: value === "parentmodule",
                    modulePath: value === "parentmodule" ? '/' : '',
                    modulePathErrmsg: "",
                    isEvent: false
                }, () => {
                    this.formValid()
                });
            } else if (value === 'event') {
                this.setState({
                    isParentModule: null,
                    isEvent: true,
                    modulePath: null,
                    modulePathErrmsg: ""
                }, () => {
                    this.formValid()
                });
            } else {
                this.setState({
                    isParentModule: false
                })
            }
        } else {
            this.setState({
                modulePath: ""
            }, () => {
                this.formValid()
            });
        }
    }

    createSystemModule = () => {
        var postData = {
            name: this.state.moduleName,
            path: this.state.modulePath,
            visibility: this.state.visibility,
            event_code: this.state.eventCode
        };

        if (this.state.selectParentID !== null) {
            postData['parentid'] = this.state.selectParentID
        }



        api.post('rest/createmodules/', postData)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `System Module created successfully`,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.hideAlertSuccess()
                })

            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to create system module`,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.hideAlertFailure()
                })
            })
    }

    hideAlertSuccess() {
        this.props.history.push(`/systemModuleSearch`)
    }

    hideAlertFailure() {
        this.setState({
            moduleName: '',
            modulePath: '',
            moduleNameErrmsg: '',
            modulePathErrmsg: '',
            isParentModule: null
        }, () => this.formValid())
    }

    getSystemModulelist = () => {
        api.get(`rest/updatemodules/`, 'hide')
            .then((response) => {
                this.setState({
                    loading: false
                })
                if (response) {
                    if (response.data.length > 0) {
                        let moduleData = [];
                        response.data.forEach((key) => {
                            if (!key.is_deleted && key.parentid === null) {
                                moduleData.push({
                                    'id': key.id,
                                    'moduleName': key.name,
                                    'modulePath': key.path,
                                    'create_date': key.create_date,
                                    'is_deleted': key.is_deleted,
                                    '': ''
                                });
                            }
                        });
                        this.setState({
                            systemModules: moduleData
                        });
                    }
                }
            })
            .catch((err) => {
                // Error handling Code
            })
    }

    render() {
        return (
            <div>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className='system-module'>
                    <div className='add'>
                        <h2>Add System Module</h2>
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
                                <option selected={true} value="show">Show</option>
                                <option value="hide">Hide</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="modulePath">
                                Type
                            </label>
                            <select onChange={(e) => this.selectModuleType(e)}>
                                <option value="">Select a Module type</option>
                                <option value="parentmodule">Parent Module</option>
                                <option value="childmodule">Child Module</option>
                                <option value="event">Event</option>
                            </select>
                        </div>

                        <div class="form-group" style={{
                            display: this.state.isEvent ? "block" : 'none'
                        }} >
                            <label for="eventCode">Event Code</label>
                            <input type="text" class="form-control" id="eventCode" value={this.state.eventCode} onChange={(e) => this.handleChange(e)} />
                            <p>{this.state.eventCodeErrmsg}</p>
                        </div>

                        <div class="form-group" style={{
                            display: this.state.isParentModule || this.state.isParentModule === null ? "none" : 'block'
                        }} >
                            <label for="modulePath">Module Path</label>
                            <input type="text" class="form-control" id="modulePath" value={this.state.modulePath} onChange={(e) => this.handleChange(e)} />
                            <p>{this.state.modulePathErrmsg}</p>
                        </div>

                        <div class="form-group" style={{
                            display: this.state.isParentModule || this.state.isParentModule === null ? "none" : 'block'
                        }} >
                            <label for="modulePath">
                                Parent Module
                                {/* <span className="hint">(Leave empty in case no Parent)</span> */}
                            </label>
                            <select onChange={(e) => this.selectParentID(e)}>
                                <option selected={this.state.selectParentID === null} value="">Select a Parent Module </option>
                                {this.state.systemModules.map(modules => (<option value={modules.id}>{modules.moduleName}</option>))}
                            </select>
                        </div>

                        <div className='action-btn'>
                            <button type="button" className="btn btn-danger" disabled={this.state.disable} onClick={() => this.createSystemModule()}>Add</button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default SystemModuleAdd;